#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LoomFormer checkpoint anatomy: weight-space spectra, circuits, degeneracies.

No data, no forward pass -- pure state_dict analysis.

  [1] per-layer circuits: QK / OV effective rank per head, FFN skeleton
      w2@w1_real, spectral norms of residual writers
  [2] w1_imag sector energy: which stream (Q/Kctx/C/U/D) the phase listens to
  [3] hidden-neuron degeneracies: duplicates, dead units, gate_proj~w1_real
  [4] depth-attn: query Gram over 60 sublayers, value-circuit ranks
  [5] embedding / final CA / aggregator spectra, LayerNorm gains
  [6] axis-stripe aggregates (layer % 3) -- carrier imprint test
  [7] anomaly flags

Usage:
  python spectra.py ckpt.pt [--init init.pt] [--device auto] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import math

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------- metrics

def spec(W: torch.Tensor):
    """SVD summary of a 2-D matrix."""
    S = torch.linalg.svdvals(W.float())
    p = S / S.sum().clamp_min(1e-12)
    erank = float(torch.exp(-(p * p.clamp_min(1e-12).log()).sum()))
    stable = float((S.pow(2).sum() / S[0].pow(2).clamp_min(1e-24)))
    return {"smax": float(S[0]), "erank": erank, "stable_rank": stable,
            "top1": float(p[0]), "full": min(W.shape),
            "erank_frac": erank / min(W.shape)}


def row_dup_stats(W: torch.Tensor, thresh_hi=0.98, thresh_lo=0.90):
    """Max off-diagonal |cosine| between rows; duplicate counts."""
    Wn = F.normalize(W.float(), dim=1, eps=1e-9)
    G = (Wn @ Wn.t()).abs()
    G.fill_diagonal_(0)
    mx = G.max(dim=1).values
    return {"cos_p50": float(mx.median()), "cos_p99": float(torch.quantile(mx, 0.99)),
            "dup98": int((mx > thresh_hi).sum()), "dup90": int((mx > thresh_lo).sum())}


def norm_stats(v: torch.Tensor):
    v = v.float()
    return {"mean": float(v.mean()), "std": float(v.std()),
            "min": float(v.min()), "max": float(v.max()),
            "near_zero": int((v.abs() < 0.05).sum())}


def rel_drift(a, b):
    a_float = a.float()
    b_float = b.to(device=a_float.device, dtype=torch.float32)
    return float((a_float - b_float).norm() / b_float.norm().clamp_min(1e-12))


# ------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--init", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    dev = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                       if args.device == "auto" else args.device)

    blob = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = {k: v.to(dev) if torch.is_tensor(v) else v for k, v in blob["model"].items()}
    sd0 = None
    if args.init:
        sd0 = torch.load(args.init, map_location="cpu", weights_only=False)["model"]
    cfg = blob["cfg"]
    L = int(cfg["layers"]); qh = int(cfg["n_q_heads"]); hd = int(cfg["head_dim"])
    N = qh * hd
    grp = cfg.get("gqa_group_size") or (qh // int(cfg["n_kv_heads"]))
    kvh = qh // int(grp)
    KV = kvh * hd
    sectors_head = str(cfg.get("phase_sectors", "head")) == "head"
    H = sd["blocks.0.ffn.w1_real.weight"].shape[0]
    HPQ = H // qh
    print(f"step={blob.get('step','?')}  L={L} d={N} H={H} qh={qh} kvh={kvh} hd={hd} "
          f"phase={'head' if sectors_head else 'open'}  device={dev}")

    out = {"layers": [], "step": blob.get("step")}

    # ---------------- [1]+[2]+[3] per-layer ----------------
    # w1_imag column layout (head sectors): [Q hd][Kctx hd][C hd][U N][D hd]
    sec_names = ["Q", "K", "C", "U", "D"]
    sec_sizes = [hd, hd, hd, N, hd] if sectors_head else [hd, N, N, N, N]
    sec_ofs = [0]
    for s in sec_sizes:
        sec_ofs.append(sec_ofs[-1] + s)

    print("\n=== [1-3] per-layer: circuits / phase sectors / degeneracies ===")
    print(" L | QK_er OV_er | ffn_er sk_smax | o_smax w2_smax | "
          "imag% Q/K/C/U/D | dup98 dead | g~w1 | drift(w1/w2/o)")
    for li in range(L):
        p = f"blocks.{li}."
        qkv = sd[p + "attn.qkv_weight"].float()
        Wq, Wk, Wv = qkv[:N], qkv[N:N + KV], qkv[N + KV:]
        O = sd[p + "attn.o.weight"].float()
        qk_er, ov_er = [], []
        for h in range(qh):
            g = h // grp
            qk = Wq[h * hd:(h + 1) * hd].t() @ Wk[g * hd:(g + 1) * hd]
            ov = O[:, h * hd:(h + 1) * hd] @ Wv[g * hd:(g + 1) * hd]
            qk_er.append(spec(qk)["erank"])
            ov_er.append(spec(ov)["erank"])
        w1 = sd[p + "ffn.w1_real.weight"].float()
        w2 = sd[p + "ffn.w2.weight"].float()
        sk = spec(w2 @ w1)
        o_s = spec(O)
        w2_s = spec(w2)
        # phase sector energy
        wi = sd[p + "ffn.w1_imag"].float()
        e = wi.pow(2)
        shares = [float(e[:, sec_ofs[i]:sec_ofs[i + 1]].sum() / e.sum())
                  for i in range(5)]
        dup = row_dup_stats(w1)
        w2c = w2.norm(dim=0)  # per hidden col
        w1r = w1.norm(dim=1)
        gain = w1r * w2c
        dead = int((gain < 0.05 * gain.median()).sum())
        gp = sd.get(p + "ffn.gate_proj.weight")
        gcos = float((F.normalize(gp.float(), dim=1) *
                      F.normalize(w1, dim=1)).sum(1).abs().mean()) if gp is not None else float("nan")
        drift = ""
        if sd0 is not None:
            drift = (f"{rel_drift(sd[p+'ffn.w1_real.weight'], sd0[p+'ffn.w1_real.weight']):.2f}/"
                     f"{rel_drift(sd[p+'ffn.w2.weight'], sd0[p+'ffn.w2.weight']):.2f}/"
                     f"{rel_drift(sd[p+'attn.o.weight'], sd0[p+'attn.o.weight']):.2f}")
        row = {"layer": li, "qk_erank_mean": sum(qk_er) / qh,
               "qk_erank_min": min(qk_er), "ov_erank_mean": sum(ov_er) / qh,
               "ov_erank_min": min(ov_er), "ffn_skeleton": sk, "o_smax": o_s["smax"],
               "w2_smax": w2_s["smax"], "imag_sector_share": dict(zip(sec_names, shares)),
               "dup": dup, "dead_hidden": dead, "gate_w1_cos": gcos}
        out["layers"].append(row)
        print(f"{li:3d} | {sum(qk_er)/qh:5.1f} {sum(ov_er)/qh:5.1f} "
              f"| {sk['erank']:6.1f} {sk['smax']:7.3f} "
              f"| {o_s['smax']:6.3f} {w2_s['smax']:7.3f} "
              f"| {'/'.join(f'{100*s:.0f}' for s in shares):>14s} "
              f"| {dup['dup98']:5d} {dead:4d} | {gcos:4.2f} | {drift}")

    # ---------------- [4] depth attention ----------------
    print("\n=== [4] depth-attn ===")
    qp = sd["depth_attn.q_params"].float()          # [2L, qh, hd]
    n_sub = qp.shape[0]
    sims = []
    for h in range(qh):
        v = F.normalize(qp[:, h], dim=1)
        G = (v @ v.t()).abs()
        G.fill_diagonal_(0)
        sims.append(G)
    Gm = torch.stack(sims).mean(0)
    print(f"  query Gram over {n_sub} sublayers (mean over heads): "
          f"offdiag |cos| mean={float(Gm.mean()):.3f} p99={float(torch.quantile(Gm.flatten(),0.99)):.3f} "
          f"pairs>0.9={int((Gm>0.9).sum())//2}")
    kvw = sd["depth_attn.kv_weight"].float()
    Wvd = kvw[N:]
    wo = sd.get("depth_attn.w_o_weight")
    if wo is not None:
        wo = wo.float()
        ers, smx = [], []
        for s in range(n_sub):
            ers.append(spec(wo[s] @ Wvd)["erank"])
            smx.append(spec(wo[s])["smax"])
        print("  value-circuit erank per sublayer:")
        print("  " + " ".join(f"{v:.0f}" for v in ers))
        print("  w_o smax per sublayer:")
        print("  " + " ".join(f"{v:.2f}" for v in smx))
        out["depth"] = {"gram_mean": float(Gm.mean()), "vc_erank": ers, "wo_smax": smx}

    # ---------------- [5] globals ----------------
    print("\n=== [5] embedding / memory head / norms ===")
    emb = sd["emb.weight"].float()
    es = spec(emb)
    mu = emb.mean(0)
    cone = float(mu.norm() / emb.norm(dim=1).mean().clamp_min(1e-12))
    rown = emb.norm(dim=1)
    lown = int((rown < 0.25 * rown.median()).sum())
    print(f"  emb: erank={es['erank']:.1f}/{es['full']} top1={es['top1']:.3f} "
          f"cone(|mean|/|row|)={cone:.3f} low-norm tokens={lown}")
    for name in ("tria_final_ca.w_qk.weight", "tria_final_ca.w_v.weight",
                 "tria_agg.up.weight", "tria_agg.reader.proj.weight",
                 "tria_agg.reader.key_proj.weight"):
        if name in sd:
            s = spec(sd[name])
            dr = f" drift={rel_drift(sd[name], sd0[name]):.3f}" if sd0 is not None and name in sd0 else ""
            print(f"  {name}: erank={s['erank']:.1f}/{s['full']} smax={s['smax']:.3f} "
                  f"top1={s['top1']:.3f}{dr}")
    gains, biases = [], []
    for li in range(L):
        for ln in ("ln_attn", "ln_ffn"):
            gains.append(sd[f"blocks.{li}.{ln}.weight"].float())
            b = sd.get(f"blocks.{li}.{ln}.bias")
            if b is not None:
                biases.append(b.float())
    g = torch.cat(gains)
    gs = norm_stats(g)
    print(f"  LN gains overall: mean={gs['mean']:.3f} std={gs['std']:.3f} "
          f"min={gs['min']:.3f} max={gs['max']:.3f} |g|<0.05: {gs['near_zero']}")
    if biases:
        b = torch.cat(biases)
        print(f"  LN biases: rms={float(b.pow(2).mean().sqrt()):.3f} absmax={float(b.abs().max()):.3f}")
    if "ln_final.weight" in sd:
        print(f"  ln_final: {norm_stats(sd['ln_final.weight'])}")

    # ---------------- [6] axis stripes ----------------
    print("\n=== [6] axis aggregates (layer % 3: 0=Rz 1=Rx 2=Ry) ===")
    for ax in range(3):
        idx = [li for li in range(L) if li % 3 == ax]
        rows = [out["layers"][li] for li in idx]
        u_share = sum(r["imag_sector_share"]["U"] for r in rows) / len(rows)
        d_share = sum(r["imag_sector_share"]["D"] for r in rows) / len(rows)
        sk = sum(r["ffn_skeleton"]["erank"] for r in rows) / len(rows)
        bias_rms = torch.stack([sd[f"blocks.{li}.ffn.w1_imag_bias"].float().pow(2).mean().sqrt()
                                for li in idx]).mean()
        wi_std = torch.stack([sd[f"blocks.{li}.ffn.w1_imag"].float().std()
                              for li in idx]).mean()
        print(f"  axis {ax}: layers={idx}")
        print(f"          U%={100*u_share:.1f} D%={100*d_share:.1f} "
              f"ffn_sk_erank={sk:.1f} imag_bias_rms={float(bias_rms):.4f} "
              f"imag_std={float(wi_std):.4f}")

    # ---------------- [7] anomaly flags ----------------
    print("\n=== [7] flags ===")
    flags = []
    for r in out["layers"]:
        li = r["layer"]
        if r["qk_erank_min"] < 0.25 * hd:
            flags.append(f"L{li}: degenerate QK head (min erank {r['qk_erank_min']:.0f}/{hd})")
        if r["ov_erank_min"] < 0.25 * hd:
            flags.append(f"L{li}: degenerate OV head (min erank {r['ov_erank_min']:.0f}/{hd})")
        if r["ffn_skeleton"]["erank_frac"] < 0.2:
            flags.append(f"L{li}: low-rank FFN skeleton ({r['ffn_skeleton']['erank']:.0f}/{N})")
        if r["dup"]["dup98"] > 0.005 * H:
            flags.append(f"L{li}: {r['dup']['dup98']} duplicate hidden units (cos>0.98)")
        if r["dead_hidden"] > 0.01 * H:
            flags.append(f"L{li}: {r['dead_hidden']} dead hidden units")
        if r["gate_w1_cos"] == r["gate_w1_cos"] and r["gate_w1_cos"] > 0.9:
            flags.append(f"L{li}: gate_proj collapsed onto w1_real (cos {r['gate_w1_cos']:.2f})")
    smaxes = torch.tensor([r["w2_smax"] for r in out["layers"]])
    z = (smaxes - smaxes.mean()) / smaxes.std().clamp_min(1e-9)
    for li in torch.nonzero(z.abs() > 3).flatten().tolist():
        flags.append(f"L{li}: w2 spectral-norm outlier ({float(smaxes[li]):.2f}, z={float(z[li]):+.1f})")
    if not flags:
        flags = ["none -- no degeneracy criteria triggered"]
    for f in flags:
        print("  *", f)
    out["flags"] = flags

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1, default=float)
        print(f"\n[json] -> {args.json}")


if __name__ == "__main__":
    main()
