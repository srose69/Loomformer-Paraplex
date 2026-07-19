#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
graph_helper.py -- registers this codebase's CUDA kernels as real
torch.library.custom_op nodes, so torch.compile/Dynamo can trace straight
through them instead of graph-breaking on every pybind/PyCapsule call.

Activated by `graph: true` in the YAML config (Config.graph, default False).
Two-phase, cost = exactly one real training step:

  Phase 1 -- apply_config() calls install_capture_hooks(sys.modules[__name__],
  tria) before the model even exists. This wraps each target class's
  forward with a one-shot recorder: no registration happens yet, nothing
  is guessed -- it just waits.

  Phase 2 -- the training loop calls finalize_registration(...) right
  after step 1's optimizer.step() (see train_one_async). By then every
  kernel's forward has been called at least once with REAL config-driven
  shapes (real batch size, real seq_len, real d_model, real carry shapes --
  whatever this specific run's YAML actually produces), and each call's
  args were copied off (as fresh zero tensors, not the live ones) into
  _captured. Registration then uses those REAL captured shapes to run the
  arity-discovery probe -- no hand-guessed placeholder shapes anywhere.
  Step 1 itself still runs on the old graph-breaking path (that IS the one
  step's cost); step 2 onward gets the registered custom_ops.

=============================================================================
WHAT'S AUTOMATIC HERE, AND WHAT ISN'T (read before extending _KERNELS)
=============================================================================
Every one of these kernels already has a correct, working
torch.autograd.Function subclass (tria.py/loomformer.py's _TriaInitFused,
_PvPowluCUDA, etc). Instead of re-deriving each one's forward/backward math
by hand a second time in torch.library's idiom, this file REPLAYS the
existing class's forward/backward UNMODIFIED, by handing them a small
duck-typed ctx object instead of a real autograd ctx:

  1. SCHEMA (arg names/types, output arity) -- arg NAMES/TYPES still have to
     be declared once per kernel (see _KERNELS below) because that
     information doesn't exist anywhere introspectable: most of these
     forward() methods have no type annotations at all, so there's nothing
     to read it off of. Output ARITY, however, is NOT declared -- it's
     discovered by literally calling the real class's forward() once (see
     point 3) and counting what comes back.

  2. THE SAVE-FOR-BACKWARD / CTX-ATTRIBUTE WIRING is derived automatically
     by _run_forward_extended(): it calls the existing forward() with tiny
     real inputs against a _CtxRecorder, then matches every tensor the
     class saved (ctx.save_for_backward(...)) back to whichever input or
     output it came from BY IDENTITY, and every plain ctx attribute
     (ctx.eps = ..., ctx.m = ...) back to whichever input arg it equals BY
     VALUE. Some kernels (tria_init/tria_step/tria_init_gate/tria_step_gate)
     save an internal tensor -- "scale" -- that their forward() saves but
     does NOT return, so identity matching against inputs/outputs alone
     can't place it; those get appended as a non-public "extra" output
     instead (public_arity tracks how many of the op's returned tensors are
     the class's OWN public outputs vs this internal plumbing -- backward
     slices grad_outputs[:public_arity] before handing off to the class's
     own .backward, which only ever expected its own arity). That match
     becomes a tiny replay "recipe" used at real setup_context() time -- so
     the actual backward formula is never re-transcribed, just the
     original class's own .backward staticmethod called directly.

  3. OUTPUT SHAPE/DTYPE is expressed directly from input shapes in each fake
     implementation. Symbolic batch/sequence dimensions stay as SymInts; fake
     tracing never executes a CUDA kernel or materializes a real probe tensor.

WHY beta_space ISN'T IN _KERNELS (but IS still wrapped): _BetaSpaceDirect's
ctx stores DERIVED values (ctx.shapes/ctx.meta are tuples built from
.shape[i] lookups across TWO different inputs, plus the 4 trailing
non-differentiable int/bool args bundled under one attribute name) --
not any single input/output tensor or a value-equal scalar arg. The
identity/equality-based recipe matcher in (2) can't represent "a tuple
built out of pieces of several inputs", so this one kernel gets its own
hand-written registrar (_register_beta_space) instead of _register_one --
same architecture (opaque forward custom_op + its own opaque backward
custom_op, since register_autograd's backward is traced by AOTAutograd and
needs the raw module call isolated exactly like every other kernel here), just
with setup_context spelled out explicitly rather than derived generically.
It still shares install_capture_hooks' real-shape-capture mechanism (see
_BETA_SPACE_TARGET) -- only the registration step itself is hand-written.

tria_final_ca: was _TriaFinalCAFused in tria.py plus a whole CUDA kernel
group (kernels/tria/tria_final_ca/) -- dead code (TriaFinalCrossAttention.
forward used a direct bmm path instead, nothing ever called .apply() on
it). Both removed entirely (class, kernel files, bindings.cpp entries) --
see the codebase's history for the removal if you need the old code back.

WHY THIS RELIES ON EXPLICIT MODULE REFERENCES, NOT import tria/import
loomformer: loomformer.py is normally run as the entry point (`python
loomformer.py --train ...`), which makes Python register the running
instance under sys.modules["__main__"] -- NOT sys.modules["loomformer"]. A
naive `import loomformer` from inside this file re-executes loomformer.py
from scratch as a SECOND, throwaway module object, and every global this
file then mutates on it (GRAPH_MODE_ENABLED, _graph_*_op) lands on that
disconnected copy, never on the real running module the training loop
actually reads from. Concretely: an earlier version of this file did
exactly that, and pvpowlu's registration silently no-op'd for this exact
reason while tria's (tria.py is never run as __main__, so `import tria`
happens to resolve to the one real module either way) worked. apply_config()
now passes sys.modules[__name__] (guaranteed to be whatever module IS
actually executing, __main__ or not) and the already-imported `tria` object
explicitly instead.
"""
import threading
from typing import Callable, Dict, List, Sequence, Tuple, Type

import torch

_LOCK = threading.Lock()

# ============================================================================
# per-kernel declarations -- the ONLY hand-written part. (op_name, module_key,
# class_name, arg_specs). module_key is "lf" or "tria", resolved against the
# module objects passed into register_all() -- never imported by name here.
# arg_specs: [(param_name, python_type), ...] matching the existing class's
# forward(ctx, *args) positional order exactly.
# ============================================================================

_KERNELS: List[Tuple[str, str, str, List[Tuple]]] = [
    ("tria_init",               "tria", "_TriaInitFused",
        [("r", torch.Tensor), ("i", torch.Tensor), ("o", torch.Tensor),
         ("alpha", float), ("axis", int)]),
    ("tria_init_gate",          "tria", "_TriaInitAndGateFused",
        [("r", torch.Tensor), ("i", torch.Tensor), ("o", torch.Tensor),
         ("w", torch.Tensor), ("alpha", float), ("axis", int)]),
    ("tria_step",               "tria", "_TriaStepFused",
        [("r", torch.Tensor), ("i", torch.Tensor), ("o", torch.Tensor),
         ("carry_prev", torch.Tensor), ("alpha", float), ("axis", int)]),
    ("tria_step_gate",          "tria", "_TriaStepAndGateFused",
        [("r", torch.Tensor), ("i", torch.Tensor), ("o", torch.Tensor),
         ("carry_prev", torch.Tensor), ("w", torch.Tensor),
         ("alpha", float), ("axis", int)]),
    ("gate_slot_mix",           "tria", "_GateSlotMixFused",
        [("carry", torch.Tensor), ("w", torch.Tensor)]),
    ("slot_attention_pool",     "tria", "_SlotAttentionPoolFused",
        [("carry", torch.Tensor), ("score_w", torch.Tensor)]),
    ("temporal_carry",          "tria", "_TemporalCarryFused",
        [("depth_carry", torch.Tensor), ("reset_mask", torch.Tensor, False)]),
    ("phase_sin",               "lf", "_PhaseSinFloorCUDA",
        [("beta", torch.Tensor), ("eps", float)]),
    ("phase_sin_secant",        "lf", "_PhaseSinSecantCUDA",
        [("beta", torch.Tensor), ("anchor", torch.Tensor, False), ("near_eps", float)]),
    ("pvpowlu",                  "lf", "_PvPowluCUDA",
        [("x1", torch.Tensor), ("x2", torch.Tensor), ("m", float)]),
    ("depth_attn",               "lf", "_DepthAttnOnlineFused",
        [("q", torch.Tensor), ("hist_k", torch.Tensor), ("hist_v", torch.Tensor)]),
    # beta_space and tria_final_ca intentionally NOT here -- see module docstring.
]

# These are NOT bugs and not coverage gaps -- confirmed by reading the
# actual call graph (see the conversation this was resolved in):
#   tria_init: tria_init_and_gate's fused top branch (-> tria_init_gate,
#     which DOES register) always returns before reaching the fallback
#     line that calls standalone tria_init(). It only executes for real if
#     that fused sibling's OWN CUDA path ever fails (extension unavailable,
#     a caught RuntimeError) -- rare, and harmless to leave unregistered
#     as long as the sibling keeps working. Not provably PERMANENTLY dead
#     (a sibling failure would make it fire), just fallback-only in the
#     observed/common case.
#   gate_slot_mix: same shadowing, TWICE (behind tria_init_gate's and
#     tria_step_gate's fused fallbacks) -- plus its own standalone
#     dispatcher function has ZERO callers anywhere else in this codebase.
# depth_attn USED TO be here (DepthAttn.forward's CUDA branch only fired
# when torch.is_tensor(hist_k), and training's parallel forward built
# history as a growing Python list) -- fixed: DepthAttn.forward now stacks
# a list into a fresh local tensor before dispatching, so the CUDA-eligible
# path is reachable during training too, not just Model.step(). It's back
# to being a normal, genuinely-tracked kernel -- if it's ever unregistered
# now, that IS worth investigating, unlike the three below.
# registration_summary() reports these separately from genuinely-missing
# kernels so "[graph] NOT registered" doesn't cry wolf about paths that
# structurally almost never execute.
_KNOWN_FALLBACK_ONLY = {"tria_init", "gate_slot_mix"}

# Config-dependent, NOT architectural like the set above: phase_sin_secant
# only ever fires (hence only ever gets captured/registered) when
# cfg.phase_grad_mode=="secant" -- under the default "floor" mode, warmup
# never exercises it at all, and treating it as permanently-required would
# make is_finalized() wait forever. loomformer.py's apply_config() calls
# set_conditionally_required() to say which mode is actually active for
# this run, so is_finalized()/registration_summary() only expect it when
# it could plausibly have fired.
_CONDITIONALLY_REQUIRED: Dict[str, bool] = {
    "phase_sin_secant": False,
    "phase_sin": True,
    "temporal_carry": False,
}


def set_conditionally_required(op_name: str, required: bool) -> None:
    if op_name not in _CONDITIONALLY_REQUIRED:
        raise ValueError(f"{op_name!r} is not a conditionally-required op -- add it to _CONDITIONALLY_REQUIRED first")
    _CONDITIONALLY_REQUIRED[op_name] = bool(required)

# beta_space needs its own capture target (same _wrap_forward_for_capture
# mechanism as _KERNELS) but a HAND-WRITTEN registrar (_register_beta_space
# below), not _register_one -- its ctx stores DERIVED shape/meta tuples
# (ctx.shapes/ctx.meta), not single input/output tensors or single-valued
# scalar attrs, which the generic identity/equality matcher can't
# represent. tria_final_ca has no such target -- it's genuinely dead code
# (nothing calls .apply() on it), unlike beta_space which is very much
# live (ParaplexFFN._beta_space's whole point).
_BETA_SPACE_TARGET = ("beta_space", "lf", "_BetaSpaceDirect")

# Registration-time-only placeholder shapes for arity discovery (see
# _register_one). Default is (2,2) for every Tensor arg; kernels whose real
# rank contract is stricter than that (carry tensors end in [...,3,3];
# depth_attn's hist_k/hist_v are [B,T,S,QH,HD]) need an explicit override so
# the ONE registration-time probe call doesn't hit an indexing error before
# any real shape is ever involved. Per-arg-name -> shape; anything not
# listed for a given op still defaults to (2, 2).
# ============================================================================
# real-shape capture -- replaces hand-guessed placeholder shapes entirely.
# install_capture_hooks() (called from apply_config(), BEFORE the model
# exists) wraps each target class's forward with a one-shot recorder; the
# FIRST real call during the actual training step -- real config, real
# batch, real model, real device -- has its args' shape/dtype/device copied
# (as a fresh zero tensor, NOT the live tensor itself -- see _capture_arg)
# into _captured. finalize_registration() (called once, right after the
# real step 1 completes) then registers every kernel that got captured
# using THOSE real shapes as the arity-discovery probe's inputs. Any kernel
# that genuinely never ran during step 1 (shouldn't happen for any of these
# 12 -- see graph_helper's usage note in loomformer.py -- but if it doesn't,
# better to leave it graph-breaking than guess) is simply left unregistered.
# ============================================================================

_captured: Dict[str, List] = {}
_hooks_installed = False
_registered_ops: set = set()  # op_names already wired up -- never re-registered


def _capture_arg(a):
    if isinstance(a, torch.Tensor):
        # Fresh zero tensor, same shape/dtype/device -- explicitly NOT the
        # real tensor: holding the live one would keep that step's entire
        # autograd graph (and all its activations) alive far past when it
        # would otherwise be freed.
        return torch.zeros_like(a).detach()
    return a


def _wrap_forward_for_capture(cls: Type[torch.autograd.Function], op_name: str) -> None:
    orig_forward = cls.forward

    def _capturing_forward(ctx, *args):
        if op_name not in _captured:
            _captured[op_name] = [_capture_arg(a) for a in args]
        return orig_forward(ctx, *args)

    cls.forward = staticmethod(_capturing_forward)


def install_capture_hooks(lf_module, tria_module) -> None:
    """Call once from apply_config(), before the model is built. Wraps every
    _KERNELS class's forward (plus beta_space's) to record real args on its
    first real call -- does NOT register anything yet (no model/data exist
    at this point to probe with)."""
    global _hooks_installed
    if _hooks_installed:
        return
    modules = {"lf": lf_module, "tria": tria_module}
    for op_name, module_key, class_name, _arg_specs in _KERNELS:
        cls = getattr(modules[module_key], class_name)
        _wrap_forward_for_capture(cls, op_name)
    bs_op_name, bs_module_key, bs_class_name = _BETA_SPACE_TARGET
    _wrap_forward_for_capture(getattr(modules[bs_module_key], bs_class_name), bs_op_name)
    _hooks_installed = True


def is_finalized() -> bool:
    """True once every _KERNELS entry (plus beta_space) EXCEPT known
    fallback-only paths (see _KNOWN_FALLBACK_ONLY) AND currently-inactive
    conditionally-required ops (see _CONDITIONALLY_REQUIRED) has been
    registered -- NOT after the first finalize_registration() call
    specifically, since calls are now incremental (see
    finalize_registration's docstring)."""
    all_names = [op_name for op_name, _, _, _ in _KERNELS] + [_BETA_SPACE_TARGET[0]]
    inactive_conditional = {n for n, active in _CONDITIONALLY_REQUIRED.items() if not active}
    required = [n for n in all_names if n not in _KNOWN_FALLBACK_ONLY and n not in inactive_conditional]
    return all(n in _registered_ops for n in required)


def registration_summary() -> Tuple[List[str], List[str], List[str]]:
    """(registered, missing, fallback_only) op names.
      registered: actually wired up (opaque custom_op calls under compile).
      missing: never fired through any finalize_registration() call yet --
        genuinely worth investigating, still on the old graph-breaking path.
      fallback_only: known to structurally almost never execute (see
        _KNOWN_FALLBACK_ONLY above), OR a conditionally-required op whose
        mode isn't active this run (see _CONDITIONALLY_REQUIRED) -- NOT
        registered is the expected, harmless state for these; don't treat
        them as a problem.
    Call after finalize_registration() to see exactly what happened,
    instead of inferring it from which warnings do or don't show up."""
    all_names = [op_name for op_name, _, _, _ in _KERNELS] + [_BETA_SPACE_TARGET[0]]
    inactive_conditional = {n for n, active in _CONDITIONALLY_REQUIRED.items() if not active}
    fallback_set = _KNOWN_FALLBACK_ONLY | inactive_conditional
    registered = [n for n in all_names if n in _registered_ops]
    missing = [n for n in all_names if n not in _registered_ops and n not in fallback_set]
    fallback_only = [n for n in all_names if n not in _registered_ops and n in fallback_set]
    return registered, missing, fallback_only


def finalize_registration(lf_module, tria_module) -> None:
    """Registers a custom_op for every kernel that's been _captured but not
    yet wired up. SAFE TO CALL MULTIPLE TIMES: each op only ever gets
    registered once (torch.library.custom_op would raise on a duplicate
    name anyway), so calling this again later just picks up whatever's
    newly appeared in _captured since the last call and leaves everything
    already-registered untouched.

    Why this needs to be callable more than once at all: some of these 12
    kernels sit behind an architectural branch (which code path the model
    actually executes -- e.g. depth_attn's tensor-history CUDA path only
    exists once enough depth history has accumulated; observed in practice
    for gate_slot_mix/slot_attention_pool/depth_attn/tria_init not all
    firing within a single one-shot synthetic warmup pass, varying run to
    run), not just a device/dtype gate a fixed synthetic warmup can force
    open. Call once before the real loop starts (cheap, catches most of
    them) AND again after the first few REAL steps (see train_one_async)
    to pick up whatever only shows up once real data/sequence positions
    actually reach that branch -- still bounded to a handful of steps, not
    "wait indefinitely"."""
    modules = {"lf": lf_module, "tria": tria_module}
    for op_name, module_key, class_name, arg_specs in _KERNELS:
        if op_name in _registered_ops or op_name not in _captured:
            continue
        cls = getattr(modules[module_key], class_name)
        op = _register_one(op_name, cls, arg_specs, _captured[op_name])
        setattr(modules[module_key], f"_graph_{op_name}_op", op)
        _registered_ops.add(op_name)

    bs_op_name, bs_module_key, bs_class_name = _BETA_SPACE_TARGET
    if bs_op_name not in _registered_ops and bs_op_name in _captured:
        _register_beta_space(modules[bs_module_key], _captured[bs_op_name])
        _registered_ops.add(bs_op_name)

    if _registered_ops:
        # Flip on as soon as ANYTHING is registered -- each dispatcher
        # already null-checks its own specific _graph_{op}_op before using
        # it, so partial registration is always safe; no need to wait for
        # every kernel before letting the ones that ARE ready take effect.
        lf_module.GRAPH_MODE_ENABLED = True
        tria_module.set_graph_mode_enabled(True)


# ============================================================================
# ctx duck-type + probe/replay engine
# ============================================================================

class _CtxRecorder:
    """Enough of torch.autograd.Function's ctx surface for an EXISTING
    forward()/backward() to run against unmodified: save_for_backward(...)
    and arbitrary attribute assignment (ctx.eps = ..., ctx.m = ...)."""
    def __init__(self):
        self.saved_tensors: Tuple[torch.Tensor, ...] = ()

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors


def _run_forward_extended(cls: Type[torch.autograd.Function], args: Sequence):
    """Runs cls.forward once for real and returns:
      - extended_out: the class's own output(s) PLUS any tensor it saved for
        backward that isn't identical to an input or a public output (e.g.
        tria's internal 'scale', which forward() saves but does NOT return
        -- so a pure input/output identity match can't place it; appending
        it as a non-public extra output is what lets it survive into our
        op's backward at all).
      - public_arity: len of the class's OWN return tuple -- callers slice
        result[:public_arity] to get exactly what the original class
        returned; everything past that is graph_helper-internal plumbing.
      - save_recipe: [("input"|"output", idx), ...] against (args,
        extended_out) -- replayed at real setup_context time.
      - attr_recipe: {ctx_attr_name: (arg_idx, python_type)} for plain
        (non-Tensor) ctx attributes, matched by value against args.
    Raises if a ctx attribute can't be traced this way -- see beta_space in
    the module docstring for the one kernel that needs a hand-written
    wrapper instead of this generic path.
    """
    ctx = _CtxRecorder()
    out = cls.forward(ctx, *args)
    out_tuple = out if isinstance(out, tuple) else (out,)
    public_arity = len(out_tuple)

    extended = list(out_tuple)
    save_recipe: List[Tuple[str, int]] = []
    for t in ctx.saved_tensors:
        found = None
        for idx, a in enumerate(args):
            if isinstance(a, torch.Tensor) and a is t:
                found = ("input", idx)
                break
        if found is None:
            for idx, o in enumerate(out_tuple):
                if o is t:
                    found = ("output", idx)
                    break
        if found is None:
            found = ("output", len(extended))
            extended.append(t)
        save_recipe.append(found)

    attr_recipe: Dict[str, Tuple[int, type]] = {}
    for name, val in vars(ctx).items():
        if name == "saved_tensors":
            continue
        found_idx = None
        for idx, a in enumerate(args):
            if not isinstance(a, torch.Tensor) and a == val:
                found_idx = idx
                break
        if found_idx is None:
            raise RuntimeError(
                f"{cls.__name__}: ctx.{name} isn't equal to any plain (non-Tensor) "
                f"input arg -- this kernel needs a hand-written graph_helper wrapper."
            )
        attr_recipe[name] = (found_idx, type(val))

    return tuple(extended), public_arity, save_recipe, attr_recipe


def _fake_outputs(meta: Tuple, device: torch.device):
    outs = tuple(torch.empty(shape, dtype=dtype, device=device) for shape, dtype in meta)
    return outs[0] if len(outs) == 1 else outs


def _symbolic_fake_meta(op_name: str, args: Sequence) -> Tuple:
    """Output shape/dtype formulas for generic custom ops; preserves SymInts."""
    tensor_args = [a for a in args if isinstance(a, torch.Tensor)]
    if not tensor_args:
        raise RuntimeError(f"{op_name}: custom op has no Tensor input")
    dtype = tensor_args[0].dtype
    if op_name == "tria_init":
        r = args[0]
        return ((tuple(r.shape) + (3, 3), dtype), ((r.numel(),), torch.float32))
    if op_name == "tria_init_gate":
        r = args[0]
        return (
            (tuple(r.shape) + (3, 3), dtype),
            (tuple(r.shape), dtype),
            ((r.numel(),), torch.float32),
        )
    if op_name == "tria_step":
        r, carry = args[0], args[3]
        return ((tuple(carry.shape), dtype), ((r.numel(),), torch.float32))
    if op_name == "tria_step_gate":
        r, carry = args[0], args[3]
        return (
            (tuple(carry.shape), dtype),
            (tuple(r.shape), dtype),
            ((r.numel(),), torch.float32),
        )
    if op_name == "gate_slot_mix":
        return ((tuple(args[0].shape[:-2]), dtype),)
    if op_name == "slot_attention_pool":
        carry = args[0]
        return (
            ((carry.shape[0], carry.shape[1], 9), dtype),
            ((carry.shape[0], carry.shape[1]), dtype),
        )
    if op_name == "temporal_carry":
        carry = args[0]
        return (
            (tuple(carry.shape), torch.float32),
            (tuple(carry.shape[:-2]), torch.float32),
        )
    if op_name in ("phase_sin", "phase_sin_secant", "pvpowlu"):
        return ((tuple(args[0].shape), dtype),)
    if op_name == "depth_attn":
        hist = args[1]  # [B,T,S,QH,HD]
        return (
            ((hist.shape[0], hist.shape[1], hist.shape[3], hist.shape[4]), dtype),
            ((hist.shape[0], hist.shape[1], hist.shape[3], hist.shape[2]), dtype),
        )
    raise RuntimeError(f"{op_name}: missing symbolic fake shape formula")


# ============================================================================
# dynamic function synthesis -- torch.library.custom_op's infer_schema reads
# REAL parameter names/annotations off the function object, so a generic
# *args signature won't work; we build one concrete function per kernel at
# registration time instead.
# ============================================================================

def _make_named_fn(arg_specs: Sequence[Tuple], arity: int, body: Callable, fn_name: str):
    names = [spec[0] for spec in arg_specs]
    ns: Dict = {"_body": body, "_TensorT": torch.Tensor}
    parts = []
    for idx, spec in enumerate(arg_specs):
        n, t = spec[:2]
        tkey = f"_t{idx}"
        ns[tkey] = t
        parts.append(f"{n}: {tkey}")
    params_src = ", ".join(parts)
    call_src = ", ".join(names)
    ret_src = "_TensorT" if arity == 1 else "tuple[" + ", ".join(["_TensorT"] * arity) + "]"
    src = f"def {fn_name}({params_src}) -> {ret_src}:\n    return _body({call_src})\n"
    exec(src, ns)  # noqa: S102 -- building a real, statically-typed op signature; see module docstring
    return ns[fn_name]


# ============================================================================
# registration
# ============================================================================

def _register_one(op_name: str, cls: Type[torch.autograd.Function],
                   arg_specs: Sequence[Tuple], placeholder_args: Sequence) -> None:
    """placeholder_args: REAL shape/dtype/device args captured from an
    actual training step (see install_capture_hooks/finalize_registration)
    -- used only to discover arity once at registration time. No guessed
    shapes anywhere in this function."""
    full_name = f"loomformer::{op_name}"

    extended0, public_arity, save_recipe0, attr_recipe0 = _run_forward_extended(cls, placeholder_args)
    arity = len(extended0)
    n_inputs = len(arg_specs)
    n_saved = len(save_recipe0)
    attr_names = sorted(attr_recipe0.keys())  # deterministic order, reused below
    input_needs_grad = [
        (spec[1] is torch.Tensor) and (len(spec) < 3 or bool(spec[2]))
        for spec in arg_specs
    ]

    # ---- forward op: opaque by construction (torch.library.custom_op's
    # whole contract), so calling cls.forward (and the raw module pybind call
    # inside it) is always safe here, compiled or not. ----
    def _body(*args):
        extended, _pub, _save, _attr = _run_forward_extended(cls, args)
        return extended if len(extended) > 1 else extended[0]

    op_fn = _make_named_fn(arg_specs, arity, _body, f"_{op_name}_op")
    op = torch.library.custom_op(full_name, mutates_args=())(op_fn)

    def _fake_body(*args):
        meta = _symbolic_fake_meta(op_name, args)
        device = next(a.device for a in args if isinstance(a, torch.Tensor))
        return _fake_outputs(meta, device)

    fake_fn = _make_named_fn(arg_specs, arity, _fake_body, f"_{op_name}_fake")
    op.register_fake(fake_fn)

    # ---- backward op: NOT opaque by default. register_autograd's backward
    # function becomes a torch.autograd.Function that AOTAutograd traces as
    # part of building the compiled joint fwd+bwd graph -- if its body calls
    # a raw module pybind function directly (like cls.backward does), that
    # trace crashes with "tensor ... not allocated yet" (confirmed against
    # a real run: phase_sin hit exactly this). Wrapping the raw call in ITS
    # OWN opaque custom_op fixes it -- same fix PyTorch's own error message
    # recommends. Schema: every saved tensor + every plain ctx attribute
    # with their original Python types + every
    # public-output grad, in that fixed order -> one Tensor per FORWARD
    # input (zero-tensor placeholder where that input wasn't a Tensor to
    # begin with -- the outer _backward below turns those back into None,
    # which custom_op schemas can't return directly).
    bwd_arg_specs = (
        [(f"saved{i}", torch.Tensor) for i in range(n_saved)]
        + [(name, attr_recipe0[name][1]) for name in attr_names]
        + [(f"grad{i}", torch.Tensor) for i in range(public_arity)]
    )

    def _bwd_body(*args):
        saved = args[:n_saved]
        attrs = args[n_saved:n_saved + len(attr_names)]
        grads = args[n_saved + len(attr_names):]
        ctx2 = _CtxRecorder()
        ctx2.saved_tensors = tuple(saved)
        for name, val in zip(attr_names, attrs):
            setattr(ctx2, name, val)
        result = cls.backward(ctx2, *grads)
        result_tuple = result if isinstance(result, tuple) else (result,)
        ref_device = grads[0].device
        return tuple(
            (result_tuple[i] if i < len(result_tuple) and result_tuple[i] is not None and input_needs_grad[i]
             else torch.zeros((), device=ref_device))
            for i in range(n_inputs)
        )

    bwd_fn = _make_named_fn(bwd_arg_specs, n_inputs, _bwd_body, f"_{op_name}_bwd_op")
    bwd_op = torch.library.custom_op(f"{full_name}_bwd", mutates_args=())(bwd_fn)

    # Precomputed ONCE, here, outside of any fake/compile context: plain
    # (shape, dtype) metadata per forward input, never the tensor object
    # itself. _bwd_fake_body below must not touch placeholder_args'
    # tensors directly (e.g. via torch.empty_like(pa)) -- doing so while
    # ALSO inside FakeTensorMode during compile tracing causes a dispatch
    # conflict between the real tensor's own mode and the active fake mode
    # (confirmed: crashes with "Multiple dispatch failed for
    # aten.empty_like -- all __torch_dispatch__ handlers returned
    # NotImplemented"). Building a fresh torch.empty(shape, dtype=...) from
    # plain metadata instead sidesteps that entirely.
    # input_meta: STATIC fallback for forward inputs that were NEVER saved
    # (so there's no live tensor at backward time to read a real shape off
    # of) -- fine for those, since backward genuinely can't need a shape
    # that varies call-to-call if it never even kept a tensor around to
    # notice the variation. It is NOT a safe stand-in for inputs that WERE
    # saved and whose shape can vary between calls to the same registered
    # op -- see below.
    input_meta = [
        (tuple(pa.shape), pa.dtype) if isinstance(pa, torch.Tensor) else None
        for pa in placeholder_args
    ]
    # Maps forward input index -> position within the saved-tensor args
    # _bwd_fake_body receives, for every input that DID get saved. Built
    # once, from save_recipe0 (already computed above via
    # _run_forward_extended), not per-call -- cheap, and save_recipe's
    # shape (which input got saved, and where) doesn't itself vary
    # between calls, even when the input's OWN tensor shape does.
    _saved_pos_by_input_idx = {idx: j for j, (kind, idx) in enumerate(save_recipe0) if kind == "input"}

    def _bwd_fake_body(*args):
        # Standard autograd contract: grad-w.r.t.-input[i] has exactly
        # input[i]'s shape/dtype for THIS call -- not necessarily the same
        # shape input[i] had back at registration time. depth_attn's
        # hist_k/hist_v are the concrete case this matters for: their S
        # (depth-history) axis genuinely grows across sub-layer-boundary
        # calls within a single forward pass (1, 2, ..., 2*LAYERS), and
        # under dynamic=None the SAME compiled graph/registered op gets
        # reused across those differing real shapes -- a fixed shape
        # captured once at registration is wrong for every S except
        # whichever one happened to be captured (confirmed: "returned an
        # invalid gradient at index 1 - got [...,1,...] but expected
        # [...,8,...]" once dynamic shape reuse made this observable).
        # Saved inputs carry their OWN live tensor right here in `args`, so
        # read shape/dtype off THAT instead of the frozen input_meta.
        ref_device = args[0].device if args else torch.device("cpu")
        outs = []
        for i, meta in enumerate(input_meta):
            if not input_needs_grad[i]:
                outs.append(torch.zeros((), device=ref_device))
                continue
            if meta is None:
                outs.append(torch.zeros((), device=ref_device))
                continue
            saved_pos = _saved_pos_by_input_idx.get(i)
            if saved_pos is not None:
                live = args[saved_pos]
                outs.append(torch.empty(live.shape, dtype=live.dtype, device=ref_device))
            else:
                shape, dtype = meta
                outs.append(torch.empty(shape, dtype=dtype, device=ref_device))
        return tuple(outs)

    bwd_fake_fn = _make_named_fn(bwd_arg_specs, n_inputs, _bwd_fake_body, f"_{op_name}_bwd_fake")
    bwd_op.register_fake(bwd_fake_fn)

    def _setup_context(ctx, inputs, output):
        output_tuple = output if isinstance(output, tuple) else (output,)
        saved = [inputs[idx] if kind == "input" else output_tuple[idx] for kind, idx in save_recipe0]
        ctx.save_for_backward(*saved)
        for name, (idx, _python_type) in attr_recipe0.items():
            setattr(ctx, name, inputs[idx])

    def _backward(ctx, *grad_outputs):
        # Only the class's OWN public outputs ever get a real gradient --
        # any extra (save-only, e.g. tria's "scale") slots this wrapper
        # appended are graph_helper-internal and never differentiated.
        grads_used = grad_outputs[:public_arity]
        attr_vals = [getattr(ctx, name) for name in attr_names]
        raw_grads = bwd_op(*ctx.saved_tensors, *attr_vals, *grads_used)
        if not isinstance(raw_grads, tuple):
            raw_grads = (raw_grads,)
        return tuple(
            raw_grads[i] if input_needs_grad[i] else None
            for i in range(n_inputs)
        )

    op.register_autograd(_backward, setup_context=_setup_context)
    op._public_arity = public_arity
    return op


def _register_beta_space(lf_module, placeholder_args) -> None:
    """Hand-written -- NOT the generic engine -- because _BetaSpaceDirect's
    ctx stores DERIVED tuples (ctx.shapes: shape[i] pulled from TWO
    different inputs; ctx.meta: the 4 trailing scalar args bundled under
    one attribute name), not single input/output tensors or single-valued
    scalar attrs. Same architecture as every other kernel here regardless:
    forward is opaque by construction (torch.library.custom_op's own
    contract); backward gets its OWN separate opaque custom_op, because
    register_autograd's backward function is NOT automatically opaque --
    it's traced by AOTAutograd while building the compiled joint graph, and
    a raw module.beta_backward_cuda call inside it crashes that trace exactly
    like it did for phase_sin before that kernel got the same treatment.
    """
    def _get_module():
        module = lf_module._try_load_cuda_beta_space()
        if module is None:
            raise RuntimeError("CUDA beta_space module is unavailable")
        return module

    @torch.library.custom_op("loomformer::beta_space", mutates_args=())
    def beta_space_op(
        u: torch.Tensor, q_h: torch.Tensor, k_ctx_h: torch.Tensor,
        c_h: torch.Tensor, d_h: torch.Tensor, w1_imag_compact: torch.Tensor,
        hidden_per_q_head: int, head_dim: int, n_q_heads: int, open_sectors: bool,
    ) -> torch.Tensor:
        module = _get_module()
        out, _r_pack, _w_contig = module.beta_forward_cuda(
            u, q_h, k_ctx_h, c_h, d_h, w1_imag_compact,
            hidden_per_q_head, head_dim, n_q_heads, open_sectors,
        )
        return out

    @beta_space_op.register_fake
    def _beta_space_fake(u, q_h, k_ctx_h, c_h, d_h, w1_imag_compact, hidden_per_q_head, head_dim, n_q_heads, open_sectors):
        B, T = u.shape[0], u.shape[1]
        hidden = w1_imag_compact.shape[0]
        return torch.empty((B, T, hidden), dtype=u.dtype, device=u.device)

    def _setup_context(ctx, inputs, output):
        u, q_h, k_ctx_h, c_h, d_h, w1_imag_compact, hidden_per_q_head, head_dim, n_q_heads, open_sectors = inputs
        ctx.save_for_backward(u, q_h, k_ctx_h, c_h, d_h, w1_imag_compact)
        ctx.shapes = (u.shape[0], u.shape[1], u.shape[2], w1_imag_compact.shape[0], w1_imag_compact.shape[1])
        ctx.meta = (hidden_per_q_head, head_dim, n_q_heads, open_sectors)

    @torch.library.custom_op("loomformer::beta_space_bwd", mutates_args=())
    def beta_space_bwd_op(
        u: torch.Tensor, q_h: torch.Tensor, k_ctx_h: torch.Tensor,
        c_h: torch.Tensor, d_h: torch.Tensor, w1_imag_compact: torch.Tensor,
        grad_out: torch.Tensor,
        hidden_per_q_head: int, head_dim: int, n_q_heads: int, open_sectors: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        grad_u, grad_q, grad_k, grad_c, grad_d, grad_w = _get_module().beta_backward_cuda_recompute(
            grad_out, u, q_h, k_ctx_h, c_h, d_h, w1_imag_compact,
            hidden_per_q_head, head_dim, n_q_heads, open_sectors,
        )
        B, T = u.shape[:2]
        QH, HD = n_q_heads, head_dim
        return (
            grad_u, grad_q.view(B, T, QH, HD), grad_k.view(B, T, QH, HD),
            grad_c.view(B, T, QH, HD), grad_d.view(B, T, QH, HD), grad_w,
        )

    @beta_space_bwd_op.register_fake
    def _beta_space_bwd_fake(u, q_h, k_ctx_h, c_h, d_h, w1_imag_compact, grad_out, hidden_per_q_head, head_dim, n_q_heads, open_sectors):
        B, T, N = u.shape
        q_shape = (B, T, n_q_heads, head_dim)
        return (
            torch.empty((B, T, N), dtype=grad_out.dtype, device=grad_out.device),
            torch.empty(q_shape, dtype=grad_out.dtype, device=grad_out.device),
            torch.empty(q_shape, dtype=grad_out.dtype, device=grad_out.device),
            torch.empty(q_shape, dtype=grad_out.dtype, device=grad_out.device),
            torch.empty(q_shape, dtype=grad_out.dtype, device=grad_out.device),
            torch.empty(w1_imag_compact.shape, dtype=grad_out.dtype, device=grad_out.device),
        )

    def _backward(ctx, grad_out):
        u, q_h, k_ctx_h, c_h, d_h, w1_imag_compact = ctx.saved_tensors
        hidden_per_q_head, head_dim, n_q_heads, open_sectors = ctx.meta
        grad_u, grad_q, grad_k, grad_c, grad_d, grad_w = beta_space_bwd_op(
            u, q_h, k_ctx_h, c_h, d_h, w1_imag_compact, grad_out.contiguous(),
            hidden_per_q_head, head_dim, n_q_heads, open_sectors,
        )
        return grad_u, grad_q, grad_k, grad_c, grad_d, grad_w, None, None, None, None

    beta_space_op.register_autograd(_backward, setup_context=_setup_context)
    lf_module._graph_beta_space_op = beta_space_op
