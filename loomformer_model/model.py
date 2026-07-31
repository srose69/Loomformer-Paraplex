from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

import tria
from inline_kernels import capped_residual, depth_history_append_pair, depth_history_init_pair
from loomformer_runtime.layouts import (
    PackedAttentionLayout, PackedChunkLayout, _unpacked_attention_layout,
    build_packed_chunk_layout, packed_layout_from_segment_ids, temporal_chunk_stops,
)
from . import state as S
from .block import Block
from .primitives import (
    RMSNorm, capped_rms, cuda_autocast_dtype_or_none,
    _fused_linear_cross_entropy_eager as fused_linear_cross_entropy_eager,
    init_embedding_fanin,
)
from .types import InferenceKVRuntime, LayerCache, TrainChunkLayerState, TriaTemporalState
from .attentions.attention_backends import (
    _attention_compute_dtype, _bf16_efficient_sdpa_supported,
    _flash_backend_cache, _probe_flash_value_fusion, _probe_te_value_fusion,
    _te_backend_cache, _try_load_cuda_packed_gather,
    _varlen_backend_failure_detail,
)
from .attentions.attention_sparse import StridedGroupedQueryCausalSelfAttention, _selected_token_layout
from .attentions.depthattn import DepthAttn

if TYPE_CHECKING:
    from loomformer import Config

class Model(nn.Module):
    def __init__(self, cfg: Config, ablation: bool = False) -> None:
        super().__init__()
        # Runtime Tria geometry has exactly one owner. Keep the Config object
        # itself (also useful for intentional live overrides in loomchat) rather
        # than copying alpha/beta/W into module or process globals.
        self.cfg = cfg
        self.ablation = bool(ablation)
        self.emb = nn.Embedding(S.VOCAB, S.N)
        active_ordinals = {
            layer: ordinal for ordinal, layer in enumerate(cfg.attn_layers)
        }
        self.blocks = nn.ModuleList([
            Block(
                active_ordinals.get(layer + 1),
                cfg.attn_token_stride,
                cfg.attn_token_schedule,
                ablation=ablation,
            )
            for layer in range(S.LAYERS)
        ])
        self.depth_attn = DepthAttn()
        self.head = nn.Linear(S.N, S.VOCAB, bias=False)
        if S.TIED_EMBEDDINGS:
            self.head.weight = self.emb.weight
        self.ln_final = RMSNorm(S.N) if S.FINAL_NORM_ENABLED else None
        self.last_tria_depth_carry: Optional[torch.Tensor] = None
        self.last_tria_document_carry: Optional[torch.Tensor] = None
        self.capture_tria_depth_carry: bool = False
        if S.TRIA_CARRY_ENABLED:
            reader = tria.SharedTriaReader(k=32)
            self.tria_agg = tria.TriaAggregator(reader, S.N)
            self.tria_final_ca = tria.TriaFinalCrossAttention(
                S.N, gamma_max=S.TRIA_GAMMA_MAX, raw_gamma_init=S.TRIA_RAW_GAMMA_INIT)
        else:
            self.tria_agg = None
            self.tria_final_ca = None
        # SFT (see loomsft.py): flip to False so refeed fires ONLY on explicit
        # <CARRY>, never on the dense W-token deadline pretrain relies on --
        # the model has no fixed-grid dependency to begin with (deadline is a
        # SAFETY fallback for raw/undocumented pretrain streams, not a learned
        # requirement), so disabling it introduces no train/inference shift.
        self.tria_hard_fire_enabled = True
        self.last_tria_document_carry_stats: Optional[dict] = None
        self.last_tria_fire_mask: Optional[torch.Tensor] = None
        self._disable_structurally_unused_params()
        self.reset_parameters()

    def _disable_structurally_unused_params(self) -> None:
        if not S.TRIA_CARRY_ENABLED or not self.blocks:
            return
        # Block 0 never receives a tria carrier input, so this gate scalar is
        # outside the training graph on every step.
        first_gate = getattr(self.blocks[0].ffn, "identity_gate", None)
        if first_gate is not None:
            first_gate.raw_alpha.requires_grad_(False)
        # The last block never emits p_out to another block, so its selector is
        # also unreachable from the loss.
        last_selector = getattr(self.blocks[-1].ffn, "gate_selector", None)
        if last_selector is not None:
            last_selector.logits.requires_grad_(False)

    def _head_in(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the optional final norm and output head."""
        return self.head(self.ln_final(x) if self.ln_final is not None else x)

    def _head_or_loss(self, h: torch.Tensor, labels: Optional[torch.Tensor],
                       ignore_index: int) -> torch.Tensor:
        """Return logits (labels=None, unchanged path) or the LM loss.

        With `FUSED_LINEAR_CE` on (Config.fused_linear_ce), the loss is
        computed via `_FusedLinearCrossEntropy`, which never materializes the
        full [B*T, VOCAB] logits tensor -- see that class's docstring.
        """
        if labels is None:
            return self._head_in(h)
        if not S.FUSED_LINEAR_CE:
            logits = self._head_in(h)
            return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1),
                                    ignore_index=ignore_index)
        hidden = self.ln_final(h) if self.ln_final is not None else h
        return fused_linear_cross_entropy_eager(
            hidden.reshape(-1, hidden.shape[-1]), self.head.weight, labels.reshape(-1),
            ignore_index, S.FUSED_LINEAR_CE_CHUNK_SIZE)

    def reset_parameters(self) -> None:
        init_embedding_fanin(self.emb)

    def _build_tria_document_reset_mask(
        self,
        idx: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        del idx  # unused now that <CARRY> no longer triggers a document reset
        reset = position_ids.eq(0)
        reset[:, 0] = True
        return reset

    def _forward_parameter_views(self):
        depth_queries = self.depth_attn.normalized_queries()
        autocast_dtype = cuda_autocast_dtype_or_none()
        if (
            depth_queries.is_cuda
            and autocast_dtype in (torch.float16, torch.bfloat16)
            and depth_queries.dtype != autocast_dtype
        ):
            depth_queries = depth_queries.to(autocast_dtype)
        gate_weights = tuple(
            (
                torch.softmax(block.ffn.gate_selector.logits, dim=-1)
                if block.ffn.gate_selector is not None
                and i + 1 < len(self.blocks)
                else None
            )
            for i, block in enumerate(self.blocks)
        )
        identity_alphas = tuple(
            (
                block.ffn.identity_gate.alpha()
                if block.ffn.identity_gate is not None and i > 0
                else None
            )
            for i, block in enumerate(self.blocks)
        )
        return depth_queries, gate_weights, identity_alphas

    def _block_core(
        self,
        block: "Block",
        h: torch.Tensor,
        hist_k: list,
        hist_v: list,
        sub_idx0: int,
        depth_queries: torch.Tensor,
        gate_weight: Optional[torch.Tensor],
        identity_alpha: Optional[torch.Tensor],
        attn_out: torch.Tensor,
        q_h: torch.Tensor,
        k_ctx_h: torch.Tensor,
        c_h: torch.Tensor,
        position_ids: torch.Tensor,
        phase_reset_mask: torch.Tensor,
        phase_trace: Optional[torch.Tensor],
        carry_prev: Optional[torch.Tensor],
        p_in: Optional[torch.Tensor],
        accT_seed: Optional[torch.Tensor],
        seed_valid: Optional[torch.Tensor],
        is_first_block: bool,
        is_last_block: bool,
    ):
        sub_idx = sub_idx0
        tria_axis = (sub_idx0 // 2) % 3
        tria_alpha = float(self.cfg.tria_carrier_alpha)
        skip, _ = self.depth_attn(
            sub_idx, hist_k, hist_v, depth_queries[sub_idx])
        if S.RESIDUAL_BRANCH_RMS_CAP is not None:
            residual = capped_residual(skip, attn_out, S.RESIDUAL_BRANCH_RMS_CAP)
        else:
            residual = skip + attn_out
        h = block.ln_attn(residual)
        kv_i = self.depth_attn.project_paired(h)
        if hist_v is None:
            hist_k = depth_history_append_pair(hist_k, kv_i)
        else:
            k_i, v_i = kv_i.unbind(dim=2)
            hist_k = torch.cat((hist_k, k_i.unsqueeze(2)), dim=2)
            hist_v = torch.cat((hist_v, v_i.unsqueeze(2)), dim=2)
        sub_idx += 1

        skip, d_h = self.depth_attn(
            sub_idx, hist_k, hist_v, depth_queries[sub_idx])
        want_tria = S.TRIA_CARRY_ENABLED and not self.ablation
        if want_tria:
            ffn_out, next_phase_trace, (r, i, o) = block.ffn(
                h, q_h, k_ctx_h, c_h, d_h,
                phase_trace=phase_trace,
                phase_reset_mask=phase_reset_mask,
                return_tria=True,
                p_in=p_in,
                identity_alpha=identity_alpha,
            )
            tria.record_depth_replay(r, i, o, tria_axis)
            if is_first_block and accT_seed is not None:
                if seed_valid is None:
                    raise ValueError("seed_valid is required with accT_seed")
                if is_last_block:
                    carry_new = tria.tria_init_seed(
                        r, i, o, accT_seed, seed_valid, axis=tria_axis,
                        alpha=tria_alpha)
                    p_out = None
                else:
                    carry_new, p_out = tria.tria_init_seed_and_gate(
                        r, i, o, accT_seed, seed_valid, gate_weight, axis=tria_axis,
                        alpha=tria_alpha)
            elif is_last_block:
                carry_new = (
                    tria.tria_init(r, i, o, axis=tria_axis, alpha=tria_alpha)
                    if carry_prev is None
                    else tria.tria_step(
                        r, i, o, carry_prev, axis=tria_axis, alpha=tria_alpha)
                )
                p_out = None
            else:
                carry_new, p_out = (
                    tria.tria_init_and_gate(
                        r, i, o, gate_weight, axis=tria_axis, alpha=tria_alpha)
                    if carry_prev is None
                    else tria.tria_step_and_gate(
                        r, i, o, carry_prev, gate_weight,
                        axis=tria_axis, alpha=tria_alpha)
                )
        else:
            ffn_out, next_phase_trace = block.ffn(
                h, q_h, k_ctx_h, c_h, d_h,
                phase_trace=phase_trace,
                phase_reset_mask=phase_reset_mask,
                identity_alpha=identity_alpha,
            )
            carry_new = carry_prev
            p_out = None

        if S.RESIDUAL_BRANCH_RMS_CAP is not None:
            residual = capped_residual(skip, ffn_out, S.RESIDUAL_BRANCH_RMS_CAP)
        else:
            residual = skip + ffn_out
        h = block.ln_ffn(residual)
        if not is_last_block:
            kv_i = self.depth_attn.project_paired(h)
            if hist_v is None:
                hist_k = depth_history_append_pair(hist_k, kv_i)
            else:
                k_i, v_i = kv_i.unbind(dim=2)
                hist_k = torch.cat((hist_k, k_i.unsqueeze(2)), dim=2)
                hist_v = torch.cat((hist_v, v_i.unsqueeze(2)), dim=2)
        return h, hist_k, hist_v, next_phase_trace, carry_new, p_out

    def _run_block(
        self,
        block: "Block",
        h: torch.Tensor,
        hist_k: list,
        hist_v: list,
        sub_idx0: int,
        depth_queries: torch.Tensor,
        gate_weight: Optional[torch.Tensor],
        identity_alpha: Optional[torch.Tensor],
        attn_mask: Optional[Any],
        position_ids: torch.Tensor,
        phase_reset_mask: torch.Tensor,
        carry_prev: Optional[torch.Tensor],
        p_in: Optional[torch.Tensor],
        inherited_context,
        selected_layout=None,
        is_last_block: bool = False,
    ):
        attn_out, q_h, k_ctx_h, c_h = block.attn(
            h, attn_mask=attn_mask, position_ids=position_ids,
            inherited_context=inherited_context, selected_layout=selected_layout)
        next_context = (q_h, k_ctx_h, c_h)
        h, hist_k, hist_v, _, carry_new, p_out = self._block_core(
            block, h, hist_k, hist_v, sub_idx0,
            depth_queries, gate_weight, identity_alpha,
            attn_out, q_h, k_ctx_h, c_h, position_ids, phase_reset_mask,
            None, carry_prev, p_in, None, None, False, is_last_block)
        return h, hist_k, hist_v, carry_new, p_out, next_context

    def _run_block_chunk(
        self,
        block: "Block",
        h: torch.Tensor,
        hist_k: list,
        hist_v: list,
        sub_idx0: int,
        depth_queries: torch.Tensor,
        gate_weight: Optional[torch.Tensor],
        identity_alpha: Optional[torch.Tensor],
        position_ids: torch.Tensor,
        phase_reset_mask: torch.Tensor,
        attention_layout: Optional[PackedAttentionLayout],
        packed_chunk: PackedChunkLayout,
        past_k_chunks: tuple,
        past_v_chunks: tuple,
        past_document_chunks: tuple,
        past_position_chunks: tuple,
        held_context,
        phase_trace: Optional[torch.Tensor],
        carry_prev: Optional[torch.Tensor],
        p_in: Optional[torch.Tensor],
        accT_seed: Optional[torch.Tensor],
        seed_valid: Optional[torch.Tensor],
        inherited_context,
        strided_chunk_layout,
        is_first_block: bool,
        is_last_block: bool,
    ):
        attn_out, q_h, k_ctx_h, c_h, k_new, v_new = block.attn.forward_chunk(
            h, past_k_chunks, past_v_chunks, position_ids, attention_layout, packed_chunk,
            inherited_context=inherited_context, held_context=held_context,
            past_document_chunks=past_document_chunks,
            past_position_chunks=past_position_chunks,
            strided_chunk_layout=strided_chunk_layout)
        next_context = (q_h, k_ctx_h, c_h)
        h, hist_k, hist_v, next_phase_trace, carry_new, p_out = self._block_core(
            block, h, hist_k, hist_v, sub_idx0,
            depth_queries, gate_weight, identity_alpha,
            attn_out, q_h, k_ctx_h, c_h, position_ids, phase_reset_mask,
            phase_trace, carry_prev, p_in, accT_seed, seed_valid,
            is_first_block, is_last_block)
        return (
            h, hist_k, hist_v, k_new, v_new, next_phase_trace,
            carry_new, p_out, next_context,
        )

    def _run_chunk_stack_impl(self, h_emb_chunk: torch.Tensor, position_ids_chunk: torch.Tensor,
                              attention_layout: Optional[PackedAttentionLayout],
                              packed_chunk: PackedChunkLayout, layer_states: list,
                              accT_seed: Optional[torch.Tensor], seed_valid: Optional[torch.Tensor],
                              endpoint_reset: torch.Tensor, want_endpoint: bool, want_tail: bool,
                              replay_tape, strided_chunk_layouts,
                              depth_queries, gate_weights, identity_alphas):
        """Run the full block stack for one temporal chunk and return tensor state."""
        n_blocks = len(self.blocks)
        kv0 = self.depth_attn.project_paired(h_emb_chunk)
        hist_k = depth_history_init_pair(kv0, 2 * n_blocks)
        if hist_k is None:
            k0, v0 = kv0.unbind(dim=2)
            hist_k = k0.unsqueeze(2)
            hist_v = v0.unsqueeze(2)
        else:
            hist_v = None
        h = h_emb_chunk
        carry = None
        p = None
        k_new_out = []
        v_new_out = []
        phase_out = []
        context_out = []
        inherited_context = None
        phase_reset_mask = position_ids_chunk.eq(0)
        with tria.depth_replay_scope(
            seed=accT_seed, seed_valid=seed_valid, tape=replay_tape
        ):
            for bi, block in enumerate(self.blocks):
                ls = layer_states[bi]
                attn = block.attn
                strided_chunk_layout = (
                    strided_chunk_layouts.get(
                        (attn.token_stride, attn.token_offset))
                    if isinstance(attn, StridedGroupedQueryCausalSelfAttention)
                    else None
                )
                h, hist_k, hist_v, k_new, v_new, next_phase_trace, carry, p, inherited_context = self._run_block_chunk(
                    block, h, hist_k, hist_v, 2 * bi,
                    depth_queries, gate_weights[bi], identity_alphas[bi],
                    position_ids_chunk,
                    phase_reset_mask,
                    attention_layout, packed_chunk,
                    ls.k_chunks, ls.v_chunks,
                    ls.document_chunks, ls.position_chunks, ls.attn_context,
                    ls.phase_trace, carry, p,
                    accT_seed, seed_valid, inherited_context,
                    strided_chunk_layout,
                    bi == 0, bi == n_blocks - 1)
                k_new_out.append(k_new)
                v_new_out.append(v_new)
                phase_out.append(next_phase_trace)
                context_out.append(tuple(x[:, -1] for x in inherited_context))
            endpoint = (
                tria.temporal_carry_endpoint(
                    carry, endpoint_reset,
                    initial_state=accT_seed,
                )
                if want_endpoint else None
            )
            tail = carry[:, -1].detach().contiguous() if want_tail else None
        return (
            h, endpoint, tail, *k_new_out, *v_new_out, *phase_out,
            *(x[0] for x in context_out),
            *(x[1] for x in context_out),
            *(x[2] for x in context_out),
        )

    @torch._dynamo.disable
    def _run_chunk_stack_checkpointed(
        self,
        h_emb_chunk: torch.Tensor,
        position_ids_chunk: torch.Tensor,
        attention_layout: Optional[PackedAttentionLayout],
        packed_chunk: PackedChunkLayout,
        layer_states: list,
        accT_seed: Optional[torch.Tensor],
        seed_valid: Optional[torch.Tensor],
        endpoint_reset: torch.Tensor,
        want_endpoint: bool,
        want_tail: bool,
        strided_chunk_layouts,
        depth_queries,
        gate_weights,
        identity_alphas,
    ):
        """Run only the activation-checkpointed stack behind a Dynamo boundary.

        Its non-reentrant checkpoint uses a per-call mutable replay tape plus
        different original/recompute Python contexts. PyTorch 2.5 Dynamo tries
        to turn checkpoint into a higher-order graph op, but does not support
        our nested ``context_fn`` and, more importantly, must not freeze the
        tape/TLS mutations into a compiled graph. Keep this correctness-
        sensitive region eager; torch.compile resumes on the tensor-only
        regions around it.
        """
        # Custom autograd ctx objects are created in the original forward,
        # while r/i/o are regenerated later. Keep the tape identity stable
        # across both passes so those ctx objects see the refilled entries.
        replay_tape = tria.new_depth_replay_tape()
        holder: dict = {}

        def context_fn():
            return (
                contextlib.nullcontext(),
                S.activation_checkpoint_recompute_context(holder),
            )

        flat = torch.utils.checkpoint.checkpoint(
            self._run_chunk_stack_impl,
            h_emb_chunk,
            position_ids_chunk,
            attention_layout,
            packed_chunk,
            layer_states,
            accT_seed,
            seed_valid,
            endpoint_reset,
            want_endpoint,
            want_tail,
            replay_tape,
            strided_chunk_layouts,
            depth_queries,
            gate_weights,
            identity_alphas,
            use_reentrant=False,
            context_fn=context_fn,
        )
        # The original pass populated the stable tape identity, but its
        # tensors must not outlive the checkpointed region. Recompute resets
        # and refills the same tape before Tria backward uses it.
        replay_tape.release_inputs()
        # Forward updates the secant EMA once. Recompute must use exactly that
        # per-layer snapshot without updating the persistent buffer again.
        holder["anchor_overrides"] = {
            id(block.ffn): block.ffn.beta_anchor.detach().clone()
            for block in self.blocks
        }
        return flat

    @torch._dynamo.disable
    def _run_chunk_stack(
        self,
        h_emb_chunk: torch.Tensor,
        position_ids_chunk: torch.Tensor,
        attention_layout: Optional[PackedAttentionLayout],
        packed_chunk: PackedChunkLayout,
        layer_states: list,
        accT_seed: Optional[torch.Tensor],
        seed_valid: Optional[torch.Tensor],
        endpoint_reset: torch.Tensor,
        want_endpoint: bool,
        want_tail: bool,
        depth_queries: torch.Tensor,
        gate_weights,
        identity_alphas,
    ):
        """Run the replay-sensitive temporal stack as one eager island."""
        n_blocks = len(self.blocks)
        strided_chunk_layouts = {}
        for i, block in enumerate(self.blocks):
            attn = block.attn
            if not isinstance(attn, StridedGroupedQueryCausalSelfAttention):
                continue
            key = (attn.token_stride, attn.token_offset)
            if key in strided_chunk_layouts:
                continue
            state = layer_states[i]
            strided_chunk_layouts[key] = attn.build_chunk_layout(
                position_ids_chunk, attention_layout, packed_chunk,
                state.document_chunks, state.position_chunks)
        if S.GRAD_CHECKPOINTING and self.training:
            flat = self._run_chunk_stack_checkpointed(
                h_emb_chunk,
                position_ids_chunk,
                attention_layout,
                packed_chunk,
                layer_states,
                accT_seed,
                seed_valid,
                endpoint_reset,
                want_endpoint,
                want_tail,
                strided_chunk_layouts,
                depth_queries,
                gate_weights,
                identity_alphas,
            )
        else:
            replay_tape = tria.new_depth_replay_tape()
            flat = self._run_chunk_stack_impl(
                h_emb_chunk, position_ids_chunk, attention_layout, packed_chunk, layer_states,
                accT_seed, seed_valid, endpoint_reset, want_endpoint, want_tail,
                replay_tape, strided_chunk_layouts,
                depth_queries, gate_weights, identity_alphas)
        h = flat[0]
        endpoint = flat[1]
        tail = flat[2]
        k_new = flat[3:3 + n_blocks]
        v_new = flat[3 + n_blocks:3 + 2 * n_blocks]
        phase = flat[3 + 2 * n_blocks:3 + 3 * n_blocks]
        q_tail = flat[3 + 3 * n_blocks:3 + 4 * n_blocks]
        k_tail = flat[3 + 4 * n_blocks:3 + 5 * n_blocks]
        c_tail = flat[3 + 5 * n_blocks:3 + 6 * n_blocks]
        sparse_layouts = [
            (
                strided_chunk_layouts[
                    (block.attn.token_stride, block.attn.token_offset)
                ].selected
                if isinstance(block.attn, StridedGroupedQueryCausalSelfAttention)
                else None
            )
            for block in self.blocks
        ]
        new_layer_states = [
            TrainChunkLayerState(
                k_chunks=layer_states[i].k_chunks + (k_new[i],),
                v_chunks=layer_states[i].v_chunks + (v_new[i],),
                phase_trace=phase[i],
                document_chunks=(
                    layer_states[i].document_chunks + (sparse_layouts[i].documents,)
                    if sparse_layouts[i] is not None
                    else layer_states[i].document_chunks
                ),
                position_chunks=(
                    layer_states[i].position_chunks + (sparse_layouts[i].positions,)
                    if sparse_layouts[i] is not None
                    else layer_states[i].position_chunks
                ),
                attn_context=(q_tail[i], k_tail[i], c_tail[i]),
            )
            for i in range(n_blocks)
        ]
        return h, endpoint, tail, new_layer_states

    def forward(self, idx: torch.Tensor, attn_mask: Optional[Any] = None,
                position_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None, ignore_index: int = -100) -> torch.Tensor:
        """Forward pass. With `labels=None` (default, all existing call sites
        unchanged) this returns full [B,T,VOCAB] logits exactly as before.
        Passing `labels` returns the scalar LM loss instead; when
        `Config.fused_linear_ce` is on, that loss is computed without ever
        materializing the full logits tensor (see `_head_or_loss`)."""
        B, T = idx.shape
        if T > S.SEQ_LEN:
            raise ValueError(f"input length {T} exceeds configured seq_len {S.SEQ_LEN}")
        # Backend probes perform a tiny backward and therefore belong outside
        # activation-checkpointed regions.  Results are cached per device/dtype.
        compute_dtype = _attention_compute_dtype(idx.device, self.emb.weight.dtype)
        if S.ATTN_IMPL in ("auto", "flash") and idx.device.type == "cuda":
            _probe_flash_value_fusion(idx.device, compute_dtype)
            idx_cuda = torch.cuda.current_device() if idx.device.index is None else int(idx.device.index)
            if not _flash_backend_cache.get((idx_cuda, compute_dtype, S.HEAD_DIM), False):
                _probe_te_value_fusion(idx.device, compute_dtype)
            major, _minor = torch.cuda.get_device_capability(idx_cuda)
            optimized = (
                _flash_backend_cache.get((idx_cuda, compute_dtype, S.HEAD_DIM), False)
                or _te_backend_cache.get((idx_cuda, compute_dtype, S.HEAD_DIM), False)
            )
            if (
                optimized
                and S.TRIA_CARRY_ENABLED
                and S.TRIA_TEMPORAL_ENABLED
                and not self.ablation
                and _try_load_cuda_packed_gather() is None
                and major >= 8
            ):
                raise RuntimeError(
                    "validated varlen attention is available, but the CUDA "
                    "packed_gather extension failed to build/load; refusing "
                    "the O(history_chunks) fallback in production mode")
            if (
                S.ATTN_IMPL == "auto"
                and major >= 8
                and compute_dtype in (torch.float16, torch.bfloat16)
                and not optimized
            ):
                raise RuntimeError(
                    "attn_impl='auto' found no validated varlen forward+backward "
                    f"backend on cuda:{idx_cuda} (SM{major}x, {compute_dtype}). "
                    f"{_varlen_backend_failure_detail(idx.device, compute_dtype)}. "
                    "Install a compatible flash-attn or Transformer Engine build; "
                    "set attn_impl='sdpa' explicitly only if the slow fallback is intentional.")
        if (
            S.ATTN_IMPL in ("auto", "sdpa")
            and S.ATTN_SDPA_COMPUTE_DTYPE == "auto"
            and idx.device.type == "cuda"
        ):
            _bf16_efficient_sdpa_supported(idx.device)
        if position_ids is None:
            position_ids = torch.arange(T, device=idx.device, dtype=torch.long).view(1, T).expand(B, T)
        else:
            position_ids = position_ids.to(device=idx.device, dtype=torch.long)
        want_chunked = S.TRIA_CARRY_ENABLED and S.TRIA_TEMPORAL_ENABLED and not self.ablation
        if not want_chunked:
            effective_attn = attn_mask
            if attn_mask is None and S.ATTN_IMPL in ("auto", "flash"):
                effective_attn = _unpacked_attention_layout(B, T, idx.device)
            return self._forward_flat(idx, attn_mask=effective_attn, position_ids=position_ids,
                                       labels=labels, ignore_index=ignore_index)
        return self._forward_chunked(idx, attn_mask=attn_mask, position_ids=position_ids,
                                      labels=labels, ignore_index=ignore_index)

    def _forward_chunked(self, idx: torch.Tensor, attn_mask: Optional[Any],
                          position_ids: torch.Tensor, labels: Optional[torch.Tensor] = None,
                          ignore_index: int = -100) -> torch.Tensor:
        B, T = idx.shape
        W = int(self.cfg.tria_temporal_window)
        h_emb = self.emb(idx)
        autocast_dtype = cuda_autocast_dtype_or_none()
        if (
            h_emb.is_cuda
            and autocast_dtype in (torch.float16, torch.bfloat16)
            and h_emb.dtype != autocast_dtype
        ):
            h_emb = h_emb.to(autocast_dtype)
        depth_queries, gate_weights, identity_alphas = (
            self._forward_parameter_views())
        document_reset = self._build_tria_document_reset_mask(idx, position_ids)
        if isinstance(attn_mask, PackedAttentionLayout):
            attention_layout = attn_mask
        elif attn_mask is None:
            attention_layout = _unpacked_attention_layout(B, T, idx.device)
        else:
            # Compatibility for old callers that still provide a dense
            # block-causal mask: recover its document layout from the already
            # authoritative reset positions, then never retain/slice the mask.
            seg = torch.cumsum(position_ids.eq(0).to(torch.int32), dim=1) - 1
            attention_layout = packed_layout_from_segment_ids(seg)
        layer_states = [TrainChunkLayerState() for _ in self.blocks]
        carry_token_id = S.CARRY_TOKEN_ID
        explicit_fire = (
            idx.eq(int(carry_token_id))
            if carry_token_id is not None
            else torch.zeros(B, T, dtype=torch.bool, device=idx.device)
        )
        if self.tria_hard_fire_enabled:
            dense_pos = torch.arange(T, device=idx.device)
            grid_pos = (dense_pos + 1).remainder(W).eq(0)
            grid_pos[-1] = False
        else:
            grid_pos = torch.zeros(T, dtype=torch.bool, device=idx.device)
        next_is_reset = torch.zeros(B, T, dtype=torch.bool, device=idx.device)
        next_is_reset[:, :-1] = document_reset[:, 1:]
        hard_fire = grid_pos.view(1, T) & ~next_is_reset
        fire_mask = hard_fire | explicit_fire
        supplied_plans = tuple(attention_layout.chunk_plans)
        supplied_fire_metadata = (
            bool(supplied_plans)
            and all(plan.ends_with_fire is not None for plan in supplied_plans)
        )
        if supplied_fire_metadata:
            boundary_positions = [
                int(plan.end) - 1
                for plan in supplied_plans
                if bool(plan.ends_with_fire)
            ]
        elif torch.compiler.is_compiling():
            boundary_positions = list(range(W - 1, T, W)) if self.tria_hard_fire_enabled else []
        else:
            boundary_positions = (grid_pos | explicit_fire.any(dim=0)).nonzero().flatten().tolist()
        boundary_set = set(boundary_positions)

        h_chunks = []
        key_carries = []
        key_depth = []
        key_valid = []
        key_positions = []
        temporal_state = None
        s = 0
        chunk_ranges: List[Tuple[int, int]] = []
        stops = (
            [int(plan.end) - 1 for plan in supplied_plans]
            if supplied_plans
            else temporal_chunk_stops(
                idx, W, self.tria_hard_fire_enabled, carry_token_id,
                compiling=torch.compiler.is_compiling())
        )
        precomputed_plans = {
            (plan.start, plan.end): plan
            for plan in attention_layout.chunk_plans
        }
        for bp in stops:
            e = min(bp + 1, T)
            if e <= s:
                continue
            chunk_ranges.append((s, e))
            packed_chunk = precomputed_plans.get((s, e))
            if (
                packed_chunk is not None
                and len(packed_chunk.piece_sizes) != len(chunk_ranges)
            ):
                # A plan depends on the complete history partition, not only
                # its current (start,end) key. Reject stale/legacy metadata
                # rather than pairing N selectors with M K/V chunks.
                packed_chunk = None
            if packed_chunk is None:
                packed_chunk = build_packed_chunk_layout(
                    attention_layout, s, e, tuple(chunk_ranges))
            seed_valid = None
            temporal_seed = None
            if temporal_state is not None:
                seed_valid = fire_mask[:, s - 1] & ~document_reset[:, s]
                temporal_seed = temporal_state
            local_reset = document_reset[:, s:e].clone()
            if seed_valid is not None:
                local_reset[:, 0] |= seed_valid
            # The temporal endpoint is consumed only if it (a) seeds the next
            # chunk (there is a next iteration, e != T) or (b) is a fire
            # boundary collected into key_carries (e-1 in boundary_set). The
            # tail chunk of a sequence is neither: grid_pos[-1]=False means it
            # never fires and there is no next chunk. Computing its endpoint
            # created a dangling autograd node whose backward never runs, so
            # its ctx (and the depth-replay tape it pins, holding r/i/o and the
            # carry graph) was never released -- leaking VRAM that grew every
            # step. Structurally verified DEAD (no path to loss), so skipping
            # it changes neither loss nor any gradient.
            endpoint_consumed = (e != T) or ((e - 1) in boundary_set)
            h_chunk, temporal_endpoint, depth_tail, layer_states = self._run_chunk_stack(
                h_emb[:, s:e], position_ids[:, s:e],
                attention_layout, packed_chunk, layer_states,
                temporal_seed, seed_valid, local_reset, endpoint_consumed,
                self.capture_tria_depth_carry,
                depth_queries, gate_weights, identity_alphas)
            if endpoint_consumed:
                temporal_state = temporal_endpoint
            h_chunks.append(h_chunk)
            if e - 1 in boundary_set:
                boundary_valid = fire_mask[:, e - 1]
                corrected_state = tria.polarm(
                    temporal_state, beta=float(self.cfg.tria_polarm_beta))
                temporal_state = torch.where(
                    boundary_valid[:, None, None, None], corrected_state, temporal_state)
                key_carries.append(temporal_state)
                if self.capture_tria_depth_carry:
                    key_depth.append(depth_tail)
                key_valid.append(boundary_valid)
                key_positions.append(e - 1)
            s = e
        if s != T:
            raise RuntimeError(f"chunk boundaries stopped at {s}, expected {T}")

        h_full = torch.cat(h_chunks, dim=1)
        self.last_tria_fire_mask = fire_mask
        self.last_tria_document_carry_stats = None
        if not key_carries:
            self.last_tria_depth_carry = None
            self.last_tria_document_carry = None
            return self._head_or_loss(h_full, labels, ignore_index)

        document_keys = torch.stack(key_carries, dim=1)
        valid_keys = torch.stack(key_valid, dim=1)
        positions = torch.tensor(key_positions, device=idx.device, dtype=torch.long)
        if self.capture_tria_depth_carry:
            self.last_tria_depth_carry = torch.stack(key_depth, dim=1).detach()
            self.last_tria_document_carry = document_keys.detach()
        else:
            self.last_tria_depth_carry = None
            self.last_tria_document_carry = None
        a_keys = self.tria_agg(document_keys)
        h_full = self.tria_final_ca(
            a_keys, h_full, attention_layout,
            carry_key_mask=valid_keys, key_positions=positions)
        return self._head_or_loss(h_full, labels, ignore_index)

    def _forward_flat(self, idx: torch.Tensor, attn_mask: Optional[Any] = None,
                       position_ids: Optional[torch.Tensor] = None,
                       labels: Optional[torch.Tensor] = None, ignore_index: int = -100) -> torch.Tensor:
        B, T = idx.shape
        h = self.emb(idx)
        n_blocks = len(self.blocks)
        kv0 = self.depth_attn.project_paired(h)
        carry = None 
        p = None      
        hist_k = depth_history_init_pair(kv0, 2 * n_blocks)
        if hist_k is None:
            k0, v0 = kv0.unbind(dim=2)
            hist_k = k0.unsqueeze(2)
            hist_v = v0.unsqueeze(2)
        else:
            hist_v = None
        depth_queries, gate_weights, identity_alphas = (
            self._forward_parameter_views())
        inherited_context = None
        selected_layouts = {}
        packed = attn_mask if isinstance(attn_mask, PackedAttentionLayout) else None
        phase_reset_mask = position_ids.eq(0)
        with tria.depth_replay_scope():
            for bi, block in enumerate(self.blocks):
                selected_layout = None
                if isinstance(block.attn, StridedGroupedQueryCausalSelfAttention):
                    key = (block.attn.token_stride, block.attn.token_offset)
                    selected_layout = selected_layouts.get(key)
                    if selected_layout is None:
                        selected_layout = _selected_token_layout(
                            position_ids, packed, *key)
                        selected_layouts[key] = selected_layout
                h, hist_k, hist_v, carry, p, inherited_context = self._run_block(
                    block, h, hist_k, hist_v, 2 * bi,
                    depth_queries, gate_weights[bi], identity_alphas[bi],
                    attn_mask, position_ids, phase_reset_mask, carry, p,
                    inherited_context, selected_layout,
                    is_last_block=bi == n_blocks - 1)
        # `carry` is the final depth-composed Tria for each token/neuron. The
        # temporal path below composes those finished 3x3 matrices over T; it
        # does not recompute r/i/o or layer-local Tria.
        self.last_tria_depth_carry = None
        self.last_tria_document_carry_stats = None
        self.last_tria_fire_mask = None
        if carry is not None:
            depth_carry = carry
            if self.capture_tria_depth_carry:
                self.last_tria_depth_carry = depth_carry.detach()
            document_reset = self._build_tria_document_reset_mask(idx, position_ids)
            document_carry = (
                tria.temporal_carry(depth_carry, document_reset)
                if S.TRIA_TEMPORAL_ENABLED
                else depth_carry
            )
            W = int(self.cfg.tria_temporal_window)
            # §6.2: hard fire at the last position of each fixed-W chunk, only
            # when the NEXT position is still the same document (a fire whose
            # next position starts a new document is meaningless -- there is
            # nothing to refeed into).
            hard_fire = ((position_ids + 1) % W == 0)
            next_is_new_document = torch.zeros_like(hard_fire)
            if T > 1:
                next_is_new_document[:, :-1] = document_reset[:, 1:]
            carry_key_mask = hard_fire & (~next_is_new_document)
            a = self.tria_agg(document_carry)
            self.last_tria_fire_mask = carry_key_mask
            if torch.compiler.is_compiling():
                self.last_tria_document_carry_stats = None
            else:
                with torch.no_grad():
                    self.last_tria_document_carry_stats = {
                        "max_abs": float(document_carry.detach().abs().amax().item()),
                        "reset_count": int(document_reset.detach().sum().item()),
                        "fire_count": int(carry_key_mask.detach().sum().item()),
                    }
            h = self.tria_final_ca(a, h, attn_mask, carry_key_mask=carry_key_mask)
        return self._head_or_loss(h, labels, ignore_index)  # [B,T,VOCAB] or scalar loss

    @torch.no_grad()
    def step(
        self,
        idx_t: torch.Tensor,
        pos_t: int,
        states,
        kv_runtime: Optional[InferenceKVRuntime] = None,
    ):
        abs_pos = int(pos_t)
        if not 0 <= abs_pos < S.SEQ_LEN:
            raise ValueError(f"abs_pos={abs_pos} is outside configured seq_len={S.SEQ_LEN}")
        is_bos = abs_pos == 0
        if states is None:
            caches, tria_ca_cache, tria_temporal_state = None, None, None
        else:
            if len(states) == 2:
                caches, tria_ca_cache = states
                tria_temporal_state = None
            else:
                caches, tria_ca_cache, tria_temporal_state = states
        if is_bos:
            caches, tria_ca_cache, tria_temporal_state = None, None, None
        if caches is None:
            caches = [LayerCache() for _ in range(S.LAYERS)]
        if tria_ca_cache is None:
            tria_ca_cache = tria.TriaCACache()
        if tria_temporal_state is None:
            tria_temporal_state = TriaTemporalState()

        h = self.emb(idx_t).view(idx_t.shape[0], 1, S.N)
        n_hist = 2 * S.LAYERS
        k0, v0 = self.depth_attn.project(h)
        hist_k = h.new_zeros(h.shape[0], h.shape[1], n_hist, S.N_Q_HEADS, S.HEAD_DIM)
        hist_v = h.new_zeros(h.shape[0], h.shape[1], n_hist, S.N_Q_HEADS, S.HEAD_DIM)
        hist_k[:, :, 0] = k0
        hist_v[:, :, 0] = v0
        sub_idx = 0
        new_caches = []
        carry = None  
        p = None      
        n_blocks = len(self.blocks)
        tria_alpha = float(self.cfg.tria_carrier_alpha)
        inherited_context = None
        # spec §12.1: seed for the (single) current token's Tria L0.
        pending = (
            torch.zeros(idx_t.shape[0], dtype=torch.bool, device=idx_t.device)
            if tria_temporal_state.refeed_pending is None
            else tria_temporal_state.refeed_pending.to(device=idx_t.device)
        )
        seed_valid = pending & (not is_bos)
        accT_seed = (
            None
            if tria_temporal_state.carry is None
            else tria_temporal_state.carry
        )
        for bi, (block, cache) in enumerate(zip(self.blocks, caches)):
            is_last_block = bi == n_blocks - 1
            attn_out, q_h, k_ctx_h, c_h, k_all, v_all, cache_pos, held_context = block.attn.step(
                h, abs_pos, cache.k, cache.v, cache.cache_pos, kv_runtime=kv_runtime,
                inherited_context=inherited_context, held_context=cache.attn_context)
            inherited_context = (q_h, k_ctx_h, c_h)
            skip, _ = self.depth_attn(sub_idx, hist_k[:, :, :sub_idx + 1], hist_v[:, :, :sub_idx + 1])
            if S.RESIDUAL_BRANCH_RMS_CAP is not None:
                skip = capped_rms(skip, S.RESIDUAL_BRANCH_RMS_CAP)
                attn_out = capped_rms(attn_out, S.RESIDUAL_BRANCH_RMS_CAP)
            h = block.ln_attn(skip + attn_out)
            k_i, v_i = self.depth_attn.project(h)
            sub_idx += 1
            hist_k[:, :, sub_idx] = k_i
            hist_v[:, :, sub_idx] = v_i

            skip, d_h = self.depth_attn(sub_idx, hist_k[:, :, :sub_idx + 1], hist_v[:, :, :sub_idx + 1])
            want_tria = S.TRIA_CARRY_ENABLED and not self.ablation
            if want_tria:
                ffn_out, next_phase_trace, (r, i, o) = block.ffn(
                    h, q_h, k_ctx_h, c_h, d_h, phase_trace=cache.phase_trace,
                    return_tria=True, p_in=p)
                if bi == 0 and accT_seed is not None:
                    accT_seed_b = accT_seed.to(device=r.device, dtype=r.dtype)
                    if is_last_block:
                        carry = tria.tria_init_seed(
                            r, i, o, accT_seed_b, seed_valid, axis=bi % 3,
                            alpha=tria_alpha)
                        p = None
                    else:
                        w = torch.softmax(block.ffn.gate_selector.logits, dim=-1)
                        carry, p = tria.tria_init_seed_and_gate(
                            r, i, o, accT_seed_b, seed_valid, w, axis=bi % 3,
                            alpha=tria_alpha)
                elif is_last_block:
                    carry = (
                        tria.tria_init(r, i, o, axis=bi % 3, alpha=tria_alpha)
                        if carry is None
                        else tria.tria_step(
                            r, i, o, carry, axis=bi % 3, alpha=tria_alpha)
                    )
                    p = None
                else:
                    w = torch.softmax(block.ffn.gate_selector.logits, dim=-1)
                    carry, p = (
                        tria.tria_init_and_gate(
                            r, i, o, w, axis=bi % 3, alpha=tria_alpha)
                        if carry is None
                        else tria.tria_step_and_gate(
                            r, i, o, carry, w, axis=bi % 3, alpha=tria_alpha)
                    )
            else:
                ffn_out, next_phase_trace = block.ffn(
                    h, q_h, k_ctx_h, c_h, d_h, phase_trace=cache.phase_trace)
                carry = None
            if S.RESIDUAL_BRANCH_RMS_CAP is not None:
                skip = capped_rms(skip, S.RESIDUAL_BRANCH_RMS_CAP)
                ffn_out = capped_rms(ffn_out, S.RESIDUAL_BRANCH_RMS_CAP)
            h = block.ln_ffn(skip + ffn_out)
            if not is_last_block:
                k_i, v_i = self.depth_attn.project(h)
                sub_idx += 1
                hist_k[:, :, sub_idx] = k_i
                hist_v[:, :, sub_idx] = v_i

            new_caches.append(LayerCache(k=k_all, v=v_all, phase_trace=next_phase_trace,
                                           cache_pos=cache_pos,
                                           cache_capacity=0 if k_all is None else k_all.shape[1],
                                           attn_context=held_context))

        self.last_tria_document_carry_stats = None
        if carry is not None:
            depth_carry_t = carry[:, 0]
            # spec §12.2: reset if this is the first ever step, a refeed was
            # pending (already consumed at L0 above), or this token is BOS.
            reset_now = pending | is_bos | (not S.TRIA_TEMPORAL_ENABLED)
            if tria_temporal_state.carry is None:
                document_carry_t = depth_carry_t
            else:
                prev_doc = tria_temporal_state.carry.to(device=h.device, dtype=depth_carry_t.dtype)
                continued = tria._local_normalize(torch.matmul(depth_carry_t, prev_doc))
                document_carry_t = torch.where(reset_now[:, None, None, None], depth_carry_t, continued)
            # spec §12.3: fire decision for the NEXT token.
            carry_token_id = S.CARRY_TOKEN_ID
            hard_fire_now = (
                self.tria_hard_fire_enabled
                and (abs_pos + 1 < S.SEQ_LEN)
                and ((abs_pos + 1) % int(self.cfg.tria_temporal_window) == 0)
            )
            if carry_token_id is not None:
                explicit_fire_now = idx_t.view(-1).eq(int(carry_token_id))
            else:
                explicit_fire_now = torch.zeros(h.shape[0], dtype=torch.bool, device=h.device)
            fire_now = explicit_fire_now | hard_fire_now
            if hard_fire_now or (
                carry_token_id is not None and bool(explicit_fire_now.any().item())
            ):
                corrected_state = tria.polarm(
                    document_carry_t, beta=float(self.cfg.tria_polarm_beta))
                document_carry_t = torch.where(
                    fire_now[:, None, None, None], corrected_state, document_carry_t)
            document_carry = document_carry_t.unsqueeze(1)
            a_t = self.tria_agg(document_carry)
            h, tria_ca_cache = self.tria_final_ca.step(
                a_t, h, tria_ca_cache, abs_pos, S.SEQ_LEN, carry_key_mask=fire_now[:, None])
            tria_temporal_state = TriaTemporalState(carry=document_carry_t, refeed_pending=fire_now)
        return self._head_in(h[:, -1, :]), (new_caches, tria_ca_cache, tria_temporal_state)

__all__ = ("Model",)
