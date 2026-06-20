from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont


def wrap_text(draw, text, font, max_width):
    lines = []
    line = ""
    for ch in text:
        test = line + ch
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width or not line:
            line = test
        else:
            lines.append(line)
            line = ch
    if line:
        lines.append(line)
    return lines


def main():
    base = Path(__file__).resolve().parent
    tex_path = base / "main.tex"
    out_path = base / "2611193_HeizoNakai_software_engineering_dai4kai.pdf"

    tex = tex_path.read_text(encoding="utf-8")

    title_match = re.search(r"\\LARGE \\textbf\{(.+?)\}", tex, re.S)
    author_match = re.search(r"\\begin\{flushright\}\s*(.+?)\s*\\end\{flushright\}", tex, re.S)
    if not title_match or not author_match:
        raise RuntimeError("Failed to parse main.tex")

    title = title_match.group(1).strip()
    author = author_match.group(1).strip()
    tail = tex.split(r"\end{flushright}", 1)[-1].split(r"\end{document}", 1)[0]
    tail = re.sub(r"\\par\\medskip\s*", " ", tail)
    tail = re.sub(r"\\[a-zA-Z@]+(?:\[[^\]]*\])?(?:\{[^{}]*\})*", " ", tail)
    tail = re.sub(r"\s+", " ", tail).strip()
    paragraphs = [tail] if tail else []

    width, height = 2480, 3508
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    font_title = ImageFont.truetype(r"C:\Windows\Fonts\YuGothB.ttc", 64)
    font_author = ImageFont.truetype(r"C:\Windows\Fonts\yumin.ttf", 38)
    font_body = ImageFont.truetype(r"C:\Windows\Fonts\yumin.ttf", 40)

    margin_x = 180
    y = 180

    bbox = draw.textbbox((0, 0), title, font=font_title)
    draw.text(((width - (bbox[2] - bbox[0])) / 2, y), title, fill="black", font=font_title)
    y += 130

    bbox = draw.textbbox((0, 0), author, font=font_author)
    draw.text((width - margin_x - (bbox[2] - bbox[0]), y), author, fill="black", font=font_author)
    y += 120

    max_width = width - margin_x * 2
    line_spacing = 16
    for para in paragraphs:
        for line in wrap_text(draw, para, font_body, max_width):
            draw.text((margin_x, y), line, fill="black", font=font_body)
            bbox = draw.textbbox((0, 0), line, font=font_body)
            y += (bbox[3] - bbox[1]) + line_spacing
        y += 30

    img.save(out_path, "PDF", resolution=300.0)
    print(out_path)


if __name__ == "__main__":
    main()
