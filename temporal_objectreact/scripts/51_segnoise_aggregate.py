#!/usr/bin/env python3
"""Aggregate the §VI per-segment outlier-dropout sweep into a CSV + LaTeX.

Reads runs produced by scripts/50_segnoise_sweep.py under
    out/results/{task_dir}/{variant}_{task}_r{rate}_segnoise/val/hard/
        {YYYYMMDD-HH-MM-SS_learnt_topological}/<episode_dirs>

Applies the configs/defaults.yaml episode blacklist and reports filtered
SR / SPL / Soft-SPL.

Outputs:
  out/section_vi_seg.csv    -- (variant, task, rate, n, ns, SR, SPL, SoftSPL)
  out/section_vi_seg.tex    -- LaTeX tables for the §VI rewrite
"""
import os, re, csv, yaml
from pathlib import Path
from collections import defaultdict

ROOT = Path(os.environ.get(
    "REPO_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
))
os.chdir(ROOT)

with open("configs/defaults.yaml") as f:
    BL = yaml.safe_load(f)["episode_blacklists"]

TS_PAT = re.compile(r"^\d{8}-\d{2}-\d{2}-\d{2}_.+")

TASKS = [
    ("imitate",  "original"),
    ("alt_goal", "alt_goal"),
    ("shortcut", "via_alt_goal"),
    ("reverse",  "original_reverse"),
]
TASK_LBL = {"imitate": "Imitate", "alt_goal": "Alt-Goal",
            "shortcut": "Shortcut", "reverse": "Reverse"}

VARIANTS = [
    ("baseline",              "Baseline (mean-pool)"),
    ("ema",                   "EMA"),
    ("gru",                   "Plain GRU"),
    ("gated_gru",             "Cosine-Gated GRU"),
    ("reliability_gated_gru", "Reliability-Gated GRU"),
]
RATES = [30, 50, 70]   # percent


def collect(ts_dir, tt):
    import numpy as np
    blk = set([*BL["all_tasks"], *BL.get(tt, [])])
    eps = [e for e in ts_dir.iterdir() if e.is_dir() and "summary" not in e.name]
    ns = 0; n_ig = 0; spls = []; soft_spls = []
    for ed in eps:
        eid = ed.name.split("__")[0] + "_"
        if eid in blk:
            n_ig += 1; continue
        m = ed / "metadata.txt"
        if not m.exists(): continue
        kv = {}
        for line in m.read_text().splitlines():
            v = line.split(":")[-1]
            if "=" in v: kv[v.split("=")[0]] = v.split("=")[1]
        if "success_status" not in kv: continue
        spd = float(kv["distance_to_final_goal_from_start"])
        rem = float(kv["final_distance"])
        soft_spl = max(0.0, 1 - rem / spd) if spd > 0 else 0.0
        if kv["success_status"] == "success":
            ns += 1
            rcsv = (ed / "results.csv").read_text().splitlines()
            x = [float(r.split(",")[1]) for r in rcsv[1:]]
            z = [float(r.split(",")[3]) for r in rcsv[1:]]
            xz = np.array(list(zip(x, z)))
            plen = np.linalg.norm(xz[1:] - xz[:-1], axis=1).sum() + rem
            spls.append(plen / max(spd, plen))
            soft_spls.append(1.0)
        else:
            soft_spls.append(soft_spl)
    den = len(eps) - n_ig
    if den <= 0: return None
    return {
        "sr": ns / den * 100.0,
        "spl": sum(spls) / den * 100.0,
        "soft_spl": sum(soft_spls) / den * 100.0,
        "n": den, "ns": ns,
    }


def find_ts(tag, tt):
    root = ROOT / "out" / "results" / tt / tag / "val" / "hard"
    if not root.is_dir(): return None
    ts = [d for d in root.iterdir() if d.is_dir() and TS_PAT.match(d.name)]
    if not ts: return None
    ts.sort(key=lambda x: x.stat().st_mtime)
    return ts[-1]


def main():
    table = defaultdict(dict)  # table[(variant, rate)][task] -> result
    n_present = 0
    for vkey, _ in VARIANTS:
        for rate in RATES:
            for task, tt in TASKS:
                tag = f"{vkey}_{task}_r{rate}_segnoise"
                ts = find_ts(tag, tt)
                if ts is None:
                    table[(vkey, rate)][task] = None
                    continue
                r = collect(ts, tt)
                table[(vkey, rate)][task] = r
                if r is not None and r.get("n", 0) >= 20:
                    n_present += 1

    print(f"[present] {n_present}/{len(VARIANTS) * len(RATES) * len(TASKS)} runs")

    # ----- CSV -----
    csv_path = ROOT / "out" / "section_vi_seg.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "rate", "task", "n", "ns", "SR", "SPL", "SoftSPL"])
        for vkey, _ in VARIANTS:
            for rate in RATES:
                for task, _ in TASKS:
                    r = table[(vkey, rate)].get(task)
                    if r is None:
                        w.writerow([vkey, rate, task, "", "", "", "", ""])
                    else:
                        w.writerow([vkey, rate, task, r["n"], r["ns"],
                                    f"{r['sr']:.2f}", f"{r['spl']:.2f}",
                                    f"{r['soft_spl']:.2f}"])
    print(f"[wrote] {csv_path}")

    # ----- Console summary -----
    print("\n=== SR sweep across per-segment outlier ratio ===")
    for rate in RATES:
        print(f"\n--- rate = 0.{rate} ---")
        hdr = f"{'Variant':30s} " + " ".join(f"{TASK_LBL[t]:>10s}" for t, _ in TASKS)
        print(hdr); print("-" * len(hdr))
        for vkey, lbl in VARIANTS:
            row = [f"{lbl:30s}"]
            for task, _ in TASKS:
                r = table[(vkey, rate)].get(task)
                row.append(f"{'--':>10s}" if r is None else f"{r['sr']:>10.2f}")
            print(" ".join(row))

    # ----- Per-rate cherry-pick best-per-task vs baseline -----
    print("\n=== CHERRY-PICK best-per-task vs baseline ===")
    for rate in RATES:
        print(f"\n--- rate = 0.{rate} ---")
        for task, _ in TASKS:
            base = table[("baseline", rate)].get(task)
            cands = []
            for vkey, lbl in VARIANTS:
                if vkey == "baseline": continue
                r = table[(vkey, rate)].get(task)
                if r is None: continue
                cands.append((vkey, lbl, r["sr"], r))
            if not cands or base is None:
                print(f"  {TASK_LBL[task]:>10s}: --"); continue
            best = max(cands, key=lambda kv: kv[2])
            d = best[2] - base["sr"]
            verdict = "WIN " if d > 0.5 else ("TIE " if abs(d) <= 0.5 else "LOSE")
            print(f"  {TASK_LBL[task]:>10s}: {best[1]:25s} "
                  f"SR={best[2]:6.2f}  vs base {base['sr']:6.2f}  "
                  f"({d:+6.2f})  {verdict}")

    # ----- LaTeX: 3 sub-tables (one per rate), full metrics -----
    tex_path = ROOT / "out" / "section_vi_seg.tex"
    with open(tex_path, "w") as f:
        f.write("% Auto-generated by 51_segnoise_aggregate.py\n")
        for rate in RATES:
            f.write(r"\begin{table*}[!htbp]" + "\n")
            f.write(r"\centering\small\setlength{\tabcolsep}{4pt}" + "\n")
            f.write(
                r"\caption{Per-segment outlier-cost dropout at $\rho{=}0."
                + str(rate)
                + r"$ (training distribution: $\rho{=}0.3$). "
                + r"Blacklist-filtered HM3D-val. Per-task best non-baseline SR in bold.}" + "\n"
            )
            f.write(r"\label{tab:vi-seg-r" + str(rate) + r"}" + "\n")
            f.write(r"\begin{tabular}{l rrrr rrrr rrrr}" + "\n")
            f.write(r"\toprule" + "\n")
            f.write(r"& \multicolumn{4}{c}{Success Rate (\%)}"
                    r" & \multicolumn{4}{c}{SPL (\%)}"
                    r" & \multicolumn{4}{c}{Soft-SPL (\%)} \\" + "\n")
            f.write(r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}" + "\n")
            f.write("Method & "
                    + " & ".join(["Imi.", "Alt.", "Sho.", "Rev."]) + " & "
                    + " & ".join(["Imi.", "Alt.", "Sho.", "Rev."]) + " & "
                    + " & ".join(["Imi.", "Alt.", "Sho.", "Rev."]) + r" \\" + "\n")
            f.write(r"\midrule" + "\n")
            # per-task best SR among non-baseline for bolding
            best_sr = {}
            for task, _ in TASKS:
                cands = []
                for vkey, _l in VARIANTS:
                    if vkey == "baseline": continue
                    r = table[(vkey, rate)].get(task)
                    if r is not None: cands.append(r["sr"])
                if cands: best_sr[task] = max(cands)
            for vkey, lbl in VARIANTS:
                cells = [lbl]
                # SR block
                for task, _ in TASKS:
                    r = table[(vkey, rate)].get(task)
                    if r is None: cells.append("--"); continue
                    s = f"{r['sr']:.2f}"
                    if vkey != "baseline" and task in best_sr \
                       and abs(r['sr'] - best_sr[task]) < 1e-6:
                        s = r"\textbf{" + s + "}"
                    cells.append(s)
                # SPL block
                for task, _ in TASKS:
                    r = table[(vkey, rate)].get(task)
                    cells.append("--" if r is None else f"{r['spl']:.2f}")
                # Soft-SPL block
                for task, _ in TASKS:
                    r = table[(vkey, rate)].get(task)
                    cells.append("--" if r is None else f"{r['soft_spl']:.2f}")
                f.write(" & ".join(cells) + r" \\" + "\n")
                if vkey == "baseline":
                    f.write(r"\midrule" + "\n")
            f.write(r"\bottomrule" + "\n")
            f.write(r"\end{tabular}" + "\n")
            f.write(r"\end{table*}" + "\n\n")
    print(f"[wrote] {tex_path}")


if __name__ == "__main__":
    main()
