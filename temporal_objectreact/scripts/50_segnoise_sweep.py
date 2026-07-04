#!/usr/bin/env python3
# §VI re-run under per-segment outlier-cost dropout (paper-aligned noise model).
# 5 variants x 4 tasks x 3 rates = 60 runs across GPUs 4-7.
#
# Variant = baseline (method=learnt, no temporal) | ema | gru | gated_gru |
#           reliability_gated_gru (method=learnt_temporal, with checkpoint).
# Rate    = 0.3 / 0.5 / 0.7 per-segment (paper trained at 0.3).
#
# Exp_name pattern: {variant}_{task}_r{rate_int}_segnoise.
import os, sys, time, subprocess, datetime

# Repo root is two levels up from this file (temporal_objectreact/scripts/).
# Override with $REPO_ROOT if running from elsewhere.
ROOT = os.environ.get(
    "REPO_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)
os.chdir(ROOT)

# Python interpreter: $PYTHON_BIN or current sys.executable.
PY = os.environ.get("PYTHON_BIN", sys.executable)
LOG = "out/dispatch_logs/segnoise_sweep.log"

# GPU ids: $SEGNOISE_GPUS="0,1,2,3" or default to all visible devices.
_gpus_env = os.environ.get("SEGNOISE_GPUS")
if _gpus_env:
    GPUS = [int(x) for x in _gpus_env.split(",") if x.strip() != ""]
else:
    GPUS = [4, 5, 6, 7]

VARIANTS = ["baseline", "ema", "gru", "gated_gru", "reliability_gated_gru"]
TASKS_ORDER = ["alt_goal", "shortcut", "imitate", "reverse"]
RATES = [0.3, 0.5, 0.7]

TASK_MAP = {
    "imitate":  ("original",     "False"),
    "alt_goal": ("alt_goal",     "False"),
    "shortcut": ("via_alt_goal", "False"),
    "reverse":  ("original",     "True"),
}

# Run order: by rate first, then by task, then by variant - so each rate
# completes broadly before the next rate starts.
QUEUE = [(v, t, r) for r in RATES for t in TASKS_ORDER for v in VARIANTS]


def now():
    return datetime.datetime.now().strftime("%H:%M:%S")


def log(msg):
    line = f"[segnoise] {now()} {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def gpu_mem(g):
    out = subprocess.check_output(
        ["nvidia-smi", f"--id={g}", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"], text=True).strip()
    return int(out)


def tag_of(variant, task, rate):
    return f"{variant}_{task}_r{int(round(rate * 100))}_segnoise"


def already_done(variant, task, rate):
    tt, rv = TASK_MAP[task]
    save_tt = "original_reverse" if rv == "True" else tt
    tag = tag_of(variant, task, rate)
    root = f"out/results/{save_tt}/{tag}/val/hard"
    if not os.path.isdir(root):
        return False
    for ts in sorted(os.listdir(root)):
        csv = os.path.join(root, ts, "results_summary.csv")
        if os.path.exists(csv):
            with open(csv) as f:
                if "Total Episodes,36" in f.read():
                    return True
    return False


def launch(g, variant, task, rate):
    tt, rv = TASK_MAP[task]
    tag = tag_of(variant, task, rate)

    # Baseline must use the original single-frame ObjectReact config
    # (gnm, goal_uses_context=False, no temporal aggregator). Temporal
    # variants use the temporal config. Pointing baseline at the temporal
    # yaml only via method=learnt would leave the temporal controller
    # config (gnm_temporal, context_size=5, goal_uses_context=True) in
    # effect and conflate baseline with the temporal setting.
    cfg = "configs/object_react.yaml" if variant == "baseline" \
          else "configs/object_react_temporal.yaml"

    cli = [PY, "-m", "temporal_objectreact.eval_runner",
           "-c", cfg, "--set",
           f"task_type={tt}", f"reverse={rv}",
           "goal_source=topological", "goal_gen.edge_weight_str=e3d_max",
           "inject_costmap_noise=true",
           f"noise_seg_ratio={rate}",
           "noise_prob=0.0",
           "noise_seed=0",
           f"exp_name={tag}",
           "start_idx=0", "end_idx=108", "step_idx=3"]

    if variant == "baseline":
        cli += ["method=learnt"]
    else:
        load = f"logs/temporal_{variant}/latest.pth"
        cli += ["method=learnt_temporal",
                f"controller.load_run={load}",
                f"controller.temporal_aggregator={variant}"]

    out = f"out/dispatch_logs/segnoise_{tag}_gpu{g}.log"
    f_out = open(out, "w")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(g)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    p = subprocess.Popen(cli, env=env, stdout=f_out, stderr=subprocess.STDOUT)
    log(f"launch {tag} on GPU{g} pid={p.pid}  -> {out}")
    return (p, tag, f_out)


def main():
    queue = list(QUEUE)
    running = {g: None for g in GPUS}
    log(f"start  total_tasks={len(queue)}  gpus={GPUS}  rates={RATES}")

    while queue or any(running[g] is not None for g in GPUS):
        for g in GPUS:
            slot = running[g]
            if slot is not None:
                p, tag, f_out = slot
                rc = p.poll()
                if rc is not None:
                    f_out.close()
                    log(f"finished {tag} on GPU{g} rc={rc}")
                    running[g] = None

            if running[g] is None and queue and gpu_mem(g) < 1000:
                v, t, r = queue[0]
                if already_done(v, t, r):
                    log(f"SKIP {tag_of(v, t, r)} (already complete)")
                    queue.pop(0)
                    continue
                queue.pop(0)
                running[g] = launch(g, v, t, r)
        time.sleep(20)

    log("all done")


if __name__ == "__main__":
    main()
