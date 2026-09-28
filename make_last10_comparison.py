"""Matched teacher/spread10/last10 comparison, generated from actual samples."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parent
current = root / "results_finetune2m_last10"
previous = root / "results_finetune2m_shared10"
columns = (
    ("Teacher: 28 blocks", current, "step0000_teacher_"),
    ("Spread10: 6000 updates", previous, "step6000_student_loop_"),
    ("Last10: 6000 updates", current, "step6000_student_loop_"),
    ("Last10: memory disabled", current, "step6000_student_no_loop_"),
)
font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
font = ImageFont.truetype(str(font_path), 24) if font_path.exists() else ImageFont.load_default()
image_size, header, row_gap = 512, 46, 38
canvas = Image.new("RGB", (len(columns) * image_size, header + 4 * (image_size + row_gap)), "white")
draw = ImageDraw.Draw(canvas)
for column, (label, folder, stem) in enumerate(columns):
    draw.text((column * image_size + 10, 10), label, fill="black", font=font)
    for case in range(4):
        path = folder / f"{stem}{case:02d}.png"
        with Image.open(path) as image:
            canvas.paste(image.convert("RGB"), (column * image_size, header + case * (image_size + row_gap)))
        draw.text((column * image_size + 10, header + case * (image_size + row_gap) + image_size + 5),
                  f"Case {case:02d}, seed {4200 + case}", fill="black", font=font)
destination = current / "comparison_teacher_spread10_last10.jpg"
canvas.save(destination, quality=93)
print(destination, destination.stat().st_size)
