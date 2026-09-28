import json
import shutil
import time
from pathlib import Path
root=Path(__file__).resolve().parent
p=Path('/tmp/sana_finetune2m/mid10_bs128_results')
h=[json.loads(x) for x in (p/'train_history.jsonl').read_text().splitlines() if x.strip()]
v=[json.loads(x) for x in (p/'validation.jsonl').read_text().splitlines() if x.strip()]
status=dict(status='cancelled_by_user',last_logged_step=h[-1]['step'],last_checkpoint_step=v[-1]['step'],
            cancelled_at=time.time(),reason='User requested fresh training from original pretrained weights')
(p/'CANCELLED.json').write_text(json.dumps(status,indent=2))
shutil.copytree(p,root/'results_finetune2m_mid10_bs128_cancelled_continuation',dirs_exist_ok=True)
print('CANCELLED_CONTINUATION_PRESERVED',status,flush=True)
