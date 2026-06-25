import { usePageImageEl } from "@/hooks/usePageImage";
import { glyphDataUri } from "@/lib/format";
import { useUiStore } from "@/store/uiStore";
import type { GlyphDTO } from "@/types/api";
import { useEffect, useRef } from "react";

interface GlyphImageProps {
  glyph: GlyphDTO;
  /** Sizing classes for the rendered <img>/<canvas> (e.g. object-contain). */
  className?: string;
}

/**
 * A glyph thumbnail that honours the grid's binarized/original view toggle
 * (see {@link useUiStore} `glyphImageMode`):
 *
 * * "binarized" — the glyph's stored foreground mask, served as a base64 PNG.
 * * "original"  — the glyph's region cropped out of the session's page image
 *   onto a canvas, so the user can compare the raw ink against the mask.
 *
 * Falls back to the binarized preview whenever the page image is unavailable
 * or still decoding, so a tile never renders blank.
 */
export function GlyphImage({ glyph, className }: GlyphImageProps) {
  const mode = useUiStore((s) => s.glyphImageMode);
  const pageImg = usePageImageEl();

  if (mode === "original" && pageImg) {
    return (
      <GlyphOriginalCanvas
        glyph={glyph}
        pageImg={pageImg}
        className={className}
      />
    );
  }

  return (
    <img
      src={glyphDataUri(glyph)}
      alt={glyph.class_name}
      className={className}
    />
  );
}

function GlyphOriginalCanvas({
  glyph,
  pageImg,
  className,
}: {
  glyph: GlyphDTO;
  pageImg: HTMLImageElement;
  className?: string;
}) {
  const ref = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // The backing buffer is the crop's pixel size, so the page region maps
    // 1:1 and CSS object-contain scales it to fit exactly like the <img>.
    ctx.drawImage(
      pageImg,
      glyph.ulx,
      glyph.uly,
      glyph.ncols,
      glyph.nrows,
      0,
      0,
      glyph.ncols,
      glyph.nrows,
    );
  }, [pageImg, glyph.ulx, glyph.uly, glyph.ncols, glyph.nrows]);

  return (
    <canvas
      ref={ref}
      width={glyph.ncols}
      height={glyph.nrows}
      aria-label={glyph.class_name}
      className={className}
    />
  );
}
