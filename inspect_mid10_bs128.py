import json
from pathlib import Path
root = Path('/cfs/cfs-85yd60mv/hongzhuyi/sana_docci_loop')
p = Path('/tmp/sana_finetune2m/mid10_bs128_results')
for name in ('train_history.jsonl', 'validation.jsonl'):
    f = p / name
    if f.exists():
        row = json.loads(f.read_text().splitlines()[-1])
        keys = ('step', 'additional_update', 'epoch', 'elapsed_sec', 'global_batch',
                'peak_allocated_gib', 'peak_reserved_gib', 'real_mse', 'subnet_no_memory_real_mse')
        print(name, {k: round(row[k], 4) if isinstance(row[k], float) else row[k] for k in keys if k in row})
print('complete', (p / 'PILOT_COMPLETE.json').exists())
for name in ('mid10_bs128_train.log', 'mid10_bs128_sync.log'):
    f = root / 'logs' / name
    if f.exists():
        text = f.read_text(errors='replace')
        if 'Traceback' in text or name.endswith('sync.log'):
            print(name, text[-1600:])
