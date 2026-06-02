import colorsys
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
SAMPLE_DIR = HERE.parent
IMAGE = SAMPLE_DIR / "NZ-Wt MSR-03 109v.png"
ANNOTATIONS = SAMPLE_DIR / "MOTHRA_NZ-Wt MSR-03 109v_annotations.json"
OUTPUT = SAMPLE_DIR / "visualization" / "NZ-Wt MSR-03 109v_annotated.png"
PREDICTED_OUTPUT = SAMPLE_DIR / "visualization" / "NZ-Wt MSR-03 109v_predicted.png"
# IMAGE = SAMPLE_DIR / "Hufnagel-example.png"
# ANNOTATIONS = SAMPLE_DIR / "Hufnagel-example_annotations.json"
# OUTPUT = SAMPLE_DIR / "visualization" / "Hufnagel-example_annotated.png"
# PREDICTED_OUTPUT = SAMPLE_DIR / "visualization" / "Hufnagel-example_predicted.png"

CLASS_COLORS = {1: "#e6194B", 2: "#3cb44b", 3: "#4363d8"}
FALLBACK_COLOR = "#f032e6"

#: Non-glyph MOTHRA classes (staff lines, stray ink). Drawn dimmed
#: on the predicted-class overlay so the eye stays on the boxes the
#: classifier actually scored.
NON_GLYPH_CLASS_IDS = {1, 3}
NON_GLYPH_DIM_COLOR = "#bbbbbb"

AXIS_COLOR = "white"
AXIS_TEXT_STROKE = "white"
MAJOR_TICK = 100
MINOR_TICK = 50


def color_for_class(class_name: str) -> str:
    """Pick a stable colour for a predicted class label.

    The hue is hashed off the label so the same class always gets the
    same colour across runs (no ad-hoc palette to maintain). Saturation
    and value are fixed so every colour stays readable on the parchment
    background.
    """
    digest = hashlib.md5(class_name.encode("utf-8")).digest()
    hue = (int.from_bytes(digest[:2], "big") % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.85)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def draw_coordinate_scheme(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    font = ImageFont.load_default()

    def label(xy, text):
        draw.text(
            xy, text, fill=AXIS_COLOR, font=font,
            stroke_width=0.2, stroke_fill=AXIS_TEXT_STROKE,
        )

    # Top edge: x-axis ticks
    for x in range(0, width + 1, MINOR_TICK):
        major = x % MAJOR_TICK == 0
        draw.line([(x, 0), (x, 10 if major else 5)], fill=AXIS_COLOR, width=1)
        if major and x > 0:
            label((x + 2, 12), str(x))

    # Left edge: y-axis ticks
    for y in range(0, height + 1, MINOR_TICK):
        major = y % MAJOR_TICK == 0
        draw.line([(0, y), (10 if major else 5, y)], fill=AXIS_COLOR, width=1)
        if major and y > 0:
            label((12, y + 2), str(y))

    # Origin marker
    label((4, 4), "(0,0)")

    # Scale bar in the lower-right corner: 100 px
    bar_len = MAJOR_TICK
    bar_y = height - 30
    bar_x_end = width - 20
    bar_x_start = bar_x_end - bar_len
    draw.line([(bar_x_start, bar_y), (bar_x_end, bar_y)], fill=AXIS_COLOR, width=3)
    draw.line([(bar_x_start, bar_y - 5), (bar_x_start, bar_y + 5)], fill=AXIS_COLOR, width=2)
    draw.line([(bar_x_end, bar_y - 5), (bar_x_end, bar_y + 5)], fill=AXIS_COLOR, width=2)
    label((bar_x_start, bar_y + 6), f"{bar_len} px")


def draw_annotation_overlay(
    image: Path = IMAGE,
    annotations: Path = ANNOTATIONS,
    output: Path = OUTPUT,
) -> None:
    """Write ``…_annotated.png``: MOTHRA classId=2 boxes only, in green."""
    data = json.loads(annotations.read_text())
    img = Image.open(image).convert("RGB")
    draw = ImageDraw.Draw(img)
    for ann in data["annotations"]:
        x, y, w, h = ann["bbox"]
        if ann["classId"] == 2:
            color = CLASS_COLORS.get(ann["classId"], FALLBACK_COLOR)
            draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
    draw_coordinate_scheme(draw, img.width, img.height)
    output.parent.mkdir(parents=True, exist_ok=True)
    img.save(output)
    print(f"wrote {output}")


def draw_prediction_overlay(
    image: Path = IMAGE,
    annotations: Path = ANNOTATIONS,
    output: Path = PREDICTED_OUTPUT,
    classified=None,
) -> None:
    """Write ``…_predicted.png``: classId=2 boxes coloured by predicted class.

    Non-glyph boxes (classId 1 and 3) are drawn dimmed for context.

    If ``classified`` is None, the function calls
    :func:`evaluate.classify_page` itself (using the same ``image`` and
    ``annotations`` paths) — convenient for the standalone CLI. Pass a
    pre-computed glyph list when a caller (e.g. ``run_pipeline.py``)
    has already run classification and wants to avoid the double-work.
    """
    if classified is None:
        # Import lazily so the annotation overlay still works in
        # environments where ic_core isn't installed.
        from evaluate import classify_page  # type: ignore[import-not-found]

        classified, _ = classify_page(page_path=image, json_path=annotations)

    data = json.loads(annotations.read_text())

    # Index predictions by glyph UUID. ingest_page preserves the
    # MOTHRA annotation `id` (minus dashes) as the Glyph UUID, so we
    # can match each prediction back to its source annotation.
    import uuid

    predicted_by_id: dict[str, tuple[str, float]] = {
        g.id: (g.class_name, g.confidence) for g in classified
    }

    img = Image.open(image).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    for ann in data["annotations"]:
        x, y, w, h = ann["bbox"]
        if ann["classId"] in NON_GLYPH_CLASS_IDS:
            draw.rectangle(
                [x, y, x + w, y + h], outline=NON_GLYPH_DIM_COLOR, width=1
            )
            continue

        key = uuid.UUID(ann["id"]).hex
        match = predicted_by_id.get(key)
        if match is None:
            # Shouldn't happen — every classId=2 box should be
            # ingested — but draw something legible if it does.
            draw.rectangle([x, y, x + w, y + h], outline=FALLBACK_COLOR, width=2)
            continue

        class_name, confidence = match
        color = color_for_class(class_name)
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        # Label above the box: class name + confidence to two places.
        label = f"{class_name} {confidence:.2f}"
        draw.text(
            (x, max(0, y - 12)),
            label,
            fill=color,
            font=font,
            stroke_width=2,
            stroke_fill="white",
        )

    draw_coordinate_scheme(draw, img.width, img.height)
    output.parent.mkdir(parents=True, exist_ok=True)
    img.save(output)
    print(f"wrote {output}")


def main():
    draw_annotation_overlay()
    draw_prediction_overlay()


if __name__ == "__main__":
    main()
