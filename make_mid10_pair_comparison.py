from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parent
pure = root / "results_finetune2m_mid10_pure"
hybrid = root / "results_finetune2m_mid10_hybrid4"
columns = (
    ("Teacher: 28 blocks", hybrid, "step0000_teacher_"),
    ("Pure train / pure ODE", pure, "step6000_student_loop_"),
    ("Pure train / hybrid ODE", pure, "step6000_hybrid_loop_"),
    ("Hybrid train / pure ODE", hybrid, "step6000_student_loop_"),
    ("Hybrid train / hybrid ODE", hybrid, "step6000_hybrid_loop_"),
)
font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
font = ImageFont.truetype(str(font_path), 22) if font_path.exists() else ImageFont.load_default()


def assemble(selected, filename):
    size, header, gap = 512, 46, 38
    canvas = Image.new("RGB", (len(selected) * size, header + 4 * (size + gap)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, (label, folder, stem) in enumerate(selected):
        draw.text((column * size + 10, 10), label, fill="black", font=font)
        for case in range(4):
            with Image.open(folder / f"{stem}{case:02d}.png") as image:
                canvas.paste(image.convert("RGB"), (column * size, header + case * (size + gap)))
            draw.text((column * size + 10, header + case * (size + gap) + size + 5),
                      f"Case {case:02d}, seed {4200 + case}", fill="black", font=font)
    path = root / filename
    canvas.save(path, quality=93)
    print(path, path.stat().st_size)


assemble(columns, "comparison_mid10_2x2.jpg")
assemble([columns[0], columns[2], columns[4]], "comparison_training_effect_hybrid_sampler.jpg")
