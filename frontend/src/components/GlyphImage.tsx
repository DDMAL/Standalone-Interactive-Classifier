import { usePageImageEl } from "@/hooks/usePageImage";
import { glyphDataUri } from "@/lib/format";
import { useUiStore } from "@/store/uiStore";
import type { GlyphDTO } from "@/types/api";
import { useEffect, useRef, useState } from "react";

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

/**
 * The `src`/`href` for one glyph, honouring the same binarized/original
 * toggle {@link GlyphImage} renders — for consumers that need a URL rather
 * than a live canvas (an SVG `<image>`, a download link, …).
 *
 * In "original" mode the glyph's region is cropped out of the page image onto
 * an offscreen canvas and exported as a data URI. That export costs a real
 * encode, so this is for the one-glyph cases (the edit panel, the split
 * canvas); the grid keeps drawing straight onto a live canvas per tile.
 *
 * Returns the binarized mask URI whenever the page image is unavailable or
 * the crop hasn't been produced yet, so the caller never renders a blank src.
 */
export function useGlyphImageSrc(glyph: GlyphDTO): string {
  const mode = useUiStore((s) => s.glyphImageMode);
  const pageImg = usePageImageEl();
  const [original, setOriginal] = useState<string | null>(null);
  const { ulx, uly, ncols, nrows } = glyph;

  useEffect(() => {
    if (mode !== "original" || !pageImg) {
      setOriginal(null);
      return;
    }
    const canvas = document.createElement("canvas");
    canvas.width = ncols;
    canvas.height = nrows;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      setOriginal(null);
      return;
    }
    ctx.drawImage(pageImg, ulx, uly, ncols, nrows, 0, 0, ncols, nrows);
    setOriginal(canvas.toDataURL("image/png"));
  }, [mode, pageImg, ulx, uly, ncols, nrows]);

  return original ?? glyphDataUri(glyph);
}
