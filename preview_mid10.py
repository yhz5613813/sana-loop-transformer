import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser()
parser.add_argument("--label", choices=("pure", "hybrid4"), required=True)
parser.add_argument("--step", type=int, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent
source = Path(f"/tmp/sana_finetune2m/mid10_{args.label}_results")
destination = root / f"results_finetune2m_mid10_{args.label}"
destination.mkdir(exist_ok=True)
columns = (("Teacher: 28 blocks", "step0000_teacher_"),
           ("Pure subnet sampling", f"step{args.step:04d}_student_loop_"),
           ("1 full + 3 subnet sampling", f"step{args.step:04d}_hybrid_loop_"),
           ("Full shared-path sampling", f"step{args.step:04d}_full_shared_"))
path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
font = ImageFont.truetype(str(path), 23) if path.exists() else ImageFont.load_default()
canvas = Image.new("RGB", (2048, 2246), "white")
draw = ImageDraw.Draw(canvas)
for column, (label, stem) in enumerate(columns):
    draw.text((column * 512 + 10, 10), label, font=font, fill="black")
    for case in range(4):
        with Image.open(source / f"{stem}{case:02d}.png") as image:
            canvas.paste(image.convert("RGB"), (column * 512, 46 + case * 550))
        draw.text((column * 512 + 10, 46 + case * 550 + 517),
                  f"{args.label}, step {args.step}, case {case}", font=font, fill="black")
path = destination / f"preview_step{args.step:04d}.jpg"
canvas.save(path, quality=93)
print(path, path.stat().st_size)
