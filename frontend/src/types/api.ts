export type ClassifierState = "import" | "classifying" | "export";

export type AnnotationFormat = "json" | "yolo";

/** How the page is binarised at ingest. See POST /sessions. */
export type BinarizationMethod = "global" | "otsu" | "sauvola";

/** Coarse MOTHRA detector category. Only "Neumes" glyphs get classified. */
export type GlyphCategory = "Text" | "Neumes" | "Staves";

/**
 * Display order for the collapsible class sections. Neumes leads because
 * that is what IC actually classifies; Text and Staves trail behind.
 */
export const CATEGORY_ORDER: GlyphCategory[] = ["Neumes", "Text", "Staves"];

/** Which sections start expanded. Only Neumes is open by default. */
export const CATEGORY_DEFAULT_OPEN: Record<GlyphCategory, boolean> = {
  Neumes: true,
  Text: false,
  Staves: false,
};

export interface GlyphDTO {
  id: string;
  class_name: string;
  confidence: number;
  id_state_manual: boolean;
  category: GlyphCategory;
  ulx: number;
  uly: number;
  ncols: number;
  nrows: number;
  image_b64: string;
}

export interface SessionDTO {
  id: string;
  state: ClassifierState;
  glyphs: GlyphDTO[];
  training_glyphs: GlyphDTO[];
  class_names: string[];
  /** Method that produced the current glyph masks. */
  binarization_method: BinarizationMethod;
  /** How many training glyphs came from a built-in preset. */
  preset_training_count: number;
  /** How many training glyphs came from an uploaded file. */
  uploaded_training_count: number;
}

export type ErrorCode =
  | "not_found"
  | "state_conflict"
  | "validation_error"
  | "deferred"
  | "internal_error";

export class ApiError extends Error {
  code: ErrorCode | "unknown";
  status: number;

  constructor(message: string, code: ApiError["code"], status: number) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
  }
}
