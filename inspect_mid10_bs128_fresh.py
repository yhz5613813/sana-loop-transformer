import json
from pathlib import Path
root=Path('/cfs/cfs-85yd60mv/hongzhuyi/sana_docci_loop')
p=Path('/tmp/sana_finetune2m/mid10_bs128_fresh_results')
for name in ('train_history.jsonl','validation.jsonl'):
    f=p/name
    if f.exists():
        row=json.loads(f.read_text().splitlines()[-1])
        keys=('step','epoch','elapsed_sec','global_batch','peak_allocated_gib','real_mse',
              'subnet_no_memory_real_mse','full_real_mse','subnet_real_mse','grad_norm')
        print(name,{k:round(row[k],4) if isinstance(row[k],float) else row[k] for k in keys if k in row})
print('complete',(p/'PILOT_COMPLETE.json').exists())
for name in ('mid10_bs128_fresh_train.log','mid10_bs128_fresh_runner.log','mid10_bs128_fresh_sync.log'):
    f=root/'logs'/name
    if f.exists():
        t=f.read_text(errors='replace')
        if 'Traceback' in t or name.endswith('sync.log'):
            print(name,t[-1400:])
