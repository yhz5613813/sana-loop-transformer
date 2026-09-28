import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
parser=argparse.ArgumentParser()
parser.add_argument('--step',type=int,default=387)
args=parser.parse_args()
root=Path(__file__).resolve().parent
source=Path('/tmp/sana_finetune2m/mid10_bs128_fresh_results')
output=root/'results_finetune2m_mid10_bs128_fresh'
output.mkdir(exist_ok=True)
try: font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
except OSError: font=ImageFont.load_default()
columns=[('Original teacher','teacher',0),('Subnet sampling','student_loop',args.step),
         ('1 full + 3 subnet','hybrid_loop',args.step),('Full shared path','full_shared',args.step)]
canvas=Image.new('RGB',(2048,2200),'white')
draw=ImageDraw.Draw(canvas)
for i in range(4):
    for j,(label,kind,step) in enumerate(columns):
        draw.text((j*512+8,i*550+8),label,fill='black',font=font)
        with Image.open(source/f'step{step:04d}_{kind}_{i:02d}.png') as im:
            canvas.paste(im.convert('RGB'),(j*512,i*550+38))
p=output/f'preview_step{args.step:04d}.jpg'
canvas.save(p,quality=94)
print(p)
