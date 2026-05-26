import colorsys
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from paths import TEST_PAGE, TEST_JSON, VIS_DIR

IMAGE = TEST_PAGE
ANNOTATIONS = TEST_JSON
OUTPUT = VIS_DIR / f"{IMAGE.stem}_annotated.png"
PREDICTED_OUTPUT = VIS_DIR / f"{IMAGE.stem}_predicted.png"

CLASS_COLORS = {1: "#e6194B", 2: "#3cb44b", 3: "#4363d8"}
FALLBACK_COLOR = "#f032e6"

#: Non-glyph MOTHRA classes (staff lines, stray ink). Drawn dimmed
#: on the predicted-class overlay so the eye stays on the boxes the
#: classifier actually scored.
NON_GLYPH_CLASS_IDS = {1, 3}
NON_GLYPH_DIM_COLOR = "#bbbbbb"

# Alpha values (0–255) control how much of the parchment shows through.
BOX_OUTLINE_ALPHA = 220
BOX_FILL_ALPHA = 55
NON_GLYPH_OUTLINE_ALPHA = 140
LABEL_BG_ALPHA = 205

AXIS_COLOR = "white"
AXIS_TEXT_STROKE = "white"
MAJOR_TICK = 100
MINOR_TICK = 50


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _with_alpha(hex_color: str, alpha: int) -> tuple[int, int, int, int]:
    r, g, b = _hex_to_rgb(hex_color)
    return r, g, b, alpha


def _draw_translucent_box(
    draw: ImageDraw.ImageDraw, x: float, y: float, w: float, h: float,
    hex_color: str, width: int = 2,
) -> None:
    draw.rectangle(
        [x, y, x + w, y + h],
        fill=_with_alpha(hex_color, BOX_FILL_ALPHA),
        outline=_with_alpha(hex_color, BOX_OUTLINE_ALPHA),
        width=width,
    )


def _draw_label_pill(
    draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont,
    x: float, y: float, text: str, hex_color: str,
) -> None:
    """Label inside a translucent coloured pill, white text on top."""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    pad_x, pad_y = 3, 1
    pill_x0 = x
    pill_y0 = max(0, y - th - 2 * pad_y - 1)
    pill_x1 = pill_x0 + tw + 2 * pad_x
    pill_y1 = pill_y0 + th + 2 * pad_y
    draw.rectangle(
        [pill_x0, pill_y0, pill_x1, pill_y1],
        fill=_with_alpha(hex_color, LABEL_BG_ALPHA),
    )
    draw.text(
        (pill_x0 + pad_x, pill_y0 + pad_y - bbox[1]),
        text,
        fill=(255, 255, 255, 255),
        font=font,
    )


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
    img = Image.open(image).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for ann in data["annotations"]:
        x, y, w, h = ann["bbox"]
        if ann["classId"] == 2:
            color = CLASS_COLORS.get(ann["classId"], FALLBACK_COLOR)
            _draw_translucent_box(draw, x, y, w, h, color)
    draw_coordinate_scheme(draw, img.width, img.height)
    composited = Image.alpha_composite(img, overlay).convert("RGB")
    output.parent.mkdir(parents=True, exist_ok=True)
    composited.save(output)
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

    img = Image.open(image).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    for ann in data["annotations"]:
        x, y, w, h = ann["bbox"]
        if ann["classId"] in NON_GLYPH_CLASS_IDS:
            draw.rectangle(
                [x, y, x + w, y + h],
                outline=_with_alpha(NON_GLYPH_DIM_COLOR, NON_GLYPH_OUTLINE_ALPHA),
                width=1,
            )
            continue

        key = uuid.UUID(ann["id"]).hex
        match = predicted_by_id.get(key)
        if match is None:
            # Shouldn't happen — every classId=2 box should be
            # ingested — but draw something legible if it does.
            _draw_translucent_box(draw, x, y, w, h, FALLBACK_COLOR)
            continue

        class_name, confidence = match
        color = color_for_class(class_name)
        _draw_translucent_box(draw, x, y, w, h, color)
        label = f"{class_name} {confidence:.2f}"
        _draw_label_pill(draw, font, x, y, label, color)

    draw_coordinate_scheme(draw, img.width, img.height)
    composited = Image.alpha_composite(img, overlay).convert("RGB")
    output.parent.mkdir(parents=True, exist_ok=True)
    composited.save(output)
    print(f"wrote {output}")


def main():
    draw_annotation_overlay()
    draw_prediction_overlay()


if __name__ == "__main__":
    main()
