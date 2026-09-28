import json
from pathlib import Path

root = Path(__file__).resolve().parent
for label in ("hybrid4", "pure"):
    folder = Path(f"/tmp/sana_finetune2m/mid10_{label}_results")
    def last(name):
        path = folder / name
        if not path.exists():
            return {}
        for line in reversed(path.read_text().splitlines()):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        return {}
    train = last("train_history.jsonl")
    val = last("validation.jsonl")
    print(label, "step", train.get("step", "waiting"), "sec", round(train.get("elapsed_sec", 0), 1),
          "val_step", val.get("step"), "scheduled_mse", round(val.get("real_mse", 0), 6),
          "subnet_no_memory_mse", round(val.get("subnet_no_memory_real_mse", 0), 6),
          "complete", (folder / "PILOT_COMPLETE.json").exists())
    log = root / "logs" / f"mid10_{label}_train.log"
    if log.exists() and "Traceback" in log.read_text():
        print("ERROR", label, log.read_text()[-1200:])
print("pair_complete", (root / "logs" / "mid10_pair_status.txt").exists())
