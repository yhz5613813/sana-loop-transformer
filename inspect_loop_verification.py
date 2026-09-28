import json
from pathlib import Path

root=Path(__file__).resolve().parent
for name in ('sequence4_bridge_smoke','control4','sequence4','sequence4_bridge'):
    p=Path('/tmp/sana_finetune2m')/f'loop_verify_{name}'
    h=p/'train_history.jsonl'
    if h.exists():
        rows=[json.loads(r) for r in h.read_text().splitlines() if r.strip()]
        v=rows[-1]
        print(name,'step',v['step'],'sec',round(v['elapsed_sec'],1),'GiB',round(v['peak_allocated_gib'],2))
    for filename in ('SMOKE_COMPLETE.json','PILOT_COMPLETE.json'):
        if (p/filename).exists():
            print(name,filename,json.loads((p/filename).read_text()))
    v=p/'validation.jsonl'
    if v.exists():
        a=json.loads(v.read_text().splitlines()[-1])
        print(name,'val',a['step'],round(a['real_mse'],5),'sub',round(a['subnet_no_memory_real_mse'],5))
    t=p/'trajectory_validation.jsonl'
    if t.exists():
        a=json.loads(t.read_text().splitlines()[-1])
        print(name,'all20',a['step'],round(a['normal']['real_mse'],5),'ownCFG',round(a['own20']['cfg_mse'],5))
for p in sorted((root/'logs').glob('loop_verify*.log')):
    s=p.read_text(errors='replace')
    if 'Traceback' in s or 'OutOfMemoryError' in s or 'RuntimeError' in s:
        print('ERROR',p.name,s[-5500:])
if (root/'logs/loop_verify_sync.log').exists():
    print((root/'logs/loop_verify_sync.log').read_text())
