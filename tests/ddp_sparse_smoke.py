import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import loomformer as lf


def run_case(device, tria_enabled):
    cfg = lf.Config(
            vocab=64,
            seq_len=8,
            batch_size=dist.get_world_size(),
            model_dim=32,
            n_q_heads=4,
            head_dim=8,
            n_kv_heads=2,
            hidden=64,
            layers=2,
            attn_layers=[1, 2],
            attn_token_stride=3,
            attn_token_schedule="staggered",
            attn_impl="sdpa",
            tria_carry_enabled=tria_enabled,
            tria_temporal_enabled=True,
            tria_temporal_auto=False,
            tria_temporal_window=2,
            amp_dtype="fp32",
        )
    lf.apply_config(cfg)
    lf.ddp_assert_config_consensus(cfg)
    torch.manual_seed(71)
    model = lf.Model(cfg).to(device)
    local_rank = int(os.environ["LOCAL_RANK"])
    ddp = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        find_unused_parameters=False,
        static_graph=True,
    )
    optimizer = torch.optim.SGD(ddp.parameters(), lr=1e-3)
    tokens = torch.tensor([[1, 2]], device=device)
    if dist.get_rank() == 0:
        segments = torch.tensor([[0, 0]], dtype=torch.int32, device=device)
    else:
        segments = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    layout = lf.packed_layout_from_segment_ids(segments)
    layout = lf.ensure_temporal_chunk_plans(tokens, layout, cfg)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        lf.ddp_sync_mutable_buffers(ddp)
        logits = ddp(
            tokens,
            attn_mask=layout,
            position_ids=layout.position_ids.long(),
        )
        logits.square().mean().backward()
        parameter = model.blocks[1].attn.qkv_weight
        if parameter.grad is None:
            raise AssertionError("empty local selector dropped QKV from the graph")
        gathered = [
            torch.empty_like(parameter.grad)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(gathered, parameter.grad)
        for other in gathered[1:]:
            torch.testing.assert_close(other, gathered[0], atol=0, rtol=0)
        if torch.count_nonzero(parameter.grad).item() == 0:
            raise AssertionError("cross-rank sparse QKV gradient vanished")
        optimizer.step()
    dist.barrier()


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl")
    try:
        rank = dist.get_rank()
        world = dist.get_world_size()
        mean, count = lf.ddp_weighted_mean(
            float((rank + 1) ** 2), rank + 1, device)
        expected_count = world * (world + 1) // 2
        expected_mean = (
            sum((index + 1) ** 2 for index in range(world))
            / expected_count
        )
        if count != expected_count or abs(mean - expected_mean) > 1e-12:
            raise AssertionError(
                f"weighted DDP reduction mismatch: {(mean, count)}")
        run_case(device, False)
        run_case(device, True)
        if dist.get_rank() == 0:
            print("DDP SPARSE SMOKE PASSED", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
