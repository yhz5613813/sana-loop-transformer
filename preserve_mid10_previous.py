import json
import shutil
import time
from pathlib import Path
root = Path(__file__).resolve().parent
fast = Path('/tmp/sana_finetune2m')
p = fast / 'mid10_pure_results'
history = [json.loads(x) for x in (p / 'train_history.jsonl').read_text().splitlines() if x.strip()]
validation = [json.loads(x) for x in (p / 'validation.jsonl').read_text().splitlines() if x.strip()]
status = dict(status='cancelled_by_user', last_logged_step=history[-1]['step'],
              last_checkpoint_step=validation[-1]['step'], cancelled_at=time.time(),
              reason='User stopped pure-subnet experiment and requested direct BS128 hybrid continuation')
(p / 'CANCELLED.json').write_text(json.dumps(status, indent=2))
(root / 'logs/mid10_pair_status.txt').write_text('PAIR_CANCELLED_BY_USER\n')
for label in ('pure', 'hybrid4'):
    shutil.copytree(fast / f'mid10_{label}_results', root / f'results_finetune2m_mid10_{label}', dirs_exist_ok=True)
print('PREVIOUS_RESULTS_PRESERVED', status, flush=True)
