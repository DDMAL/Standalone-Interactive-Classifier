import { useUiStore } from "@/store/uiStore";
import {
  type ReactNode,
  createContext,
  useContext,
  useEffect,
  useState,
} from "react";

const PageImageContext = createContext<HTMLImageElement | null>(null);

/**
 * Loads the session's original page image once and shares the decoded element
 * with every glyph tile. The grid's "original" view crops each glyph's region
 * out of this single element onto a canvas, so we never refetch the page per
 * glyph. Resolves to `null` until the image has decoded (or when the session
 * has no page), in which case tiles fall back to the binarized preview.
 */
export function PageImageProvider({ children }: { children: ReactNode }) {
  const url = useUiStore((s) => s.pageObjectUrl);
  const [img, setImg] = useState<HTMLImageElement | null>(null);

  useEffect(() => {
    if (!url) {
      setImg(null);
      return;
    }
    // Drop the previous page's element while the new one decodes so a session
    // switch can't briefly crop glyphs out of the wrong page.
    setImg(null);
    const im = new Image();
    im.decoding = "async";
    let alive = true;
    im.onload = () => {
      if (alive) setImg(im);
    };
    im.src = url;
    return () => {
      alive = false;
      im.onload = null;
    };
  }, [url]);

  return (
    <PageImageContext.Provider value={img}>
      {children}
    </PageImageContext.Provider>
  );
}

/** The decoded original page image, or `null` until it has loaded / if absent. */
export function usePageImageEl(): HTMLImageElement | null {
  return useContext(PageImageContext);
}
