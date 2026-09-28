"""Assemble matched teacher, 24-block, and shared-10 image samples."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


root = Path(__file__).resolve().parent
current = root / "results_finetune2m_shared10"
previous = root / "results_finetune2m_trainable"
columns = (
    ("Sana teacher (28 blocks)", current, "step0000_teacher_"),
    ("Previous student (24 blocks, 1200 steps)", previous, "step1200_student_loop_"),
    ("Shared student (10 blocks, 6000 steps)", current, "step6000_student_loop_"),
)
font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
font = ImageFont.truetype(str(font_path), 22) if font_path.exists() else ImageFont.load_default()
width, image_size, header, row_gap = 1536, 512, 46, 38
canvas = Image.new("RGB", (width, header + 4 * (image_size + row_gap)), "white")
draw = ImageDraw.Draw(canvas)
for column, (label, folder, stem) in enumerate(columns):
    draw.text((column * image_size + 10, 10), label, fill="black", font=font)
    for case in range(4):
        path = folder / f"{stem}{case:02d}.png"
        with Image.open(path) as image:
            canvas.paste(image.convert("RGB"),
                         (column * image_size, header + case * (image_size + row_gap)))
        draw.text((column * image_size + 10,
                   header + case * (image_size + row_gap) + image_size + 5),
                  f"Case {case:02d}", fill="black", font=font)
destination = current / "comparison_teacher_24layer_shared10.jpg"
canvas.save(destination, quality=93)
print(destination, destination.stat().st_size)
