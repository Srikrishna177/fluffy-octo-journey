"""Prints the final reproduction-vs-extension comparison numbers, computed
directly from the logged JSONL/JSON files (no hand-copied numbers)."""
import json
import os

EXT_DIR = os.path.dirname(__file__)
REPRO_LOGS = os.path.join(EXT_DIR, "..", "..", "logs")
EXT_LOGS = os.path.join(EXT_DIR, "..", "logs")


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iters(rows):
    return [r for r in rows if r.get("event") == "iter"]


def main():
    base_rows = iters(load_jsonl(os.path.join(REPRO_LOGS, "curves", "ddpo_train.jsonl")))
    ext_rows = iters(load_jsonl(os.path.join(EXT_LOGS, "curves", "ddpo_internal_train.jsonl")))

    base_last = base_rows[-1]
    ext_last = ext_rows[-1]

    base_kl_rows = [r for r in base_rows if "kl_to_base" in r]
    ext_kl_rows = [r for r in ext_rows if "kl_to_base" in r]

    with open(os.path.join(REPRO_LOGS, "eval_before_after.json")) as f:
        base_eval = json.load(f)
    ext_eval_path = os.path.join(EXT_LOGS, "eval_external_judge.json")
    ext_eval = None
    if os.path.exists(ext_eval_path):
        with open(ext_eval_path) as f:
            ext_eval = json.load(f)

    print("=== Reproduction (external reward) ===")
    print(f"iters={len(base_rows)} elapsed_sec={base_last['elapsed_sec']} "
          f"final_reward_mean={base_last['reward_mean']} final_reward_std={base_last['reward_std']}")
    print(f"final KL-to-base (last logged): {base_kl_rows[-1]}")
    print(f"eval_before_after.json: {json.dumps(base_eval)}")

    print("\n=== Extension (internal-probe reward) ===")
    print(f"iters={len(ext_rows)} elapsed_sec={ext_last['elapsed_sec']} "
          f"final_reward_mean={ext_last['reward_mean']} final_reward_std={ext_last['reward_std']}")
    if ext_kl_rows:
        print(f"final KL-to-base (last logged): {ext_kl_rows[-1]}")
    if ext_eval:
        print(f"eval_external_judge.json: {json.dumps(ext_eval)}")
    else:
        print("eval_external_judge.json: NOT YET GENERATED")

    # matched-elapsed-time KL comparison
    print("\n=== KL-to-base at matched elapsed time ===")
    for target in [30, 60, 120, 300, 600, 900, 1150]:
        b = min(base_kl_rows, key=lambda r: abs(r["elapsed_sec"] - target)) if base_kl_rows else None
        e = min(ext_kl_rows, key=lambda r: abs(r["elapsed_sec"] - target)) if ext_kl_rows else None
        b_s = f"elapsed={b['elapsed_sec']} kl={b['kl_to_base']} reward={b['reward_mean']}" if b else "n/a"
        e_s = f"elapsed={e['elapsed_sec']} kl={e['kl_to_base']} reward={e['reward_mean']}" if e else "n/a"
        print(f"~{target}s | baseline: {b_s} | extension: {e_s}")


if __name__ == "__main__":
    main()
