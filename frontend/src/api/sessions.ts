import { http, postForBlob } from "@/api/client";
import type {
  AnnotationFormat,
  GlyphCategory,
  GlyphDTO,
  SessionDTO,
} from "@/types/api";

export interface CreateSessionArgs {
  pageImage: File;
  annotations: File;
  annotationsFormat: AnnotationFormat;
  classNames?: string[];
  /** Filename of a pre-built training set (see {@link listTrainingSets}). */
  trainingXml?: string;
}

export function createSession(args: CreateSessionArgs): Promise<SessionDTO> {
  const form = new FormData();
  form.append("page_image", args.pageImage);
  form.append("annotations", args.annotations);
  form.append("annotations_format", args.annotationsFormat);
  if (args.classNames && args.classNames.length > 0) {
    form.append("class_names", JSON.stringify(args.classNames));
  }
  if (args.trainingXml) {
    form.append("training_xml", args.trainingXml);
  }
  return http.postForm<SessionDTO>("/sessions", form);
}

/** List the pre-built training-set XML filenames under core/data/derived. */
export const listTrainingSets = () =>
  http.get<string[]>("/training-sets");

export const getSession = (id: string) =>
  http.get<SessionDTO>(`/sessions/${id}`);

export const deleteSession = (id: string) => http.delete(`/sessions/${id}`);

export const classify = (id: string, k = 1) =>
  http.post<SessionDTO>(`/sessions/${id}/classify`, { k });

export interface UpdateGlyphArgs {
  class_name?: string | null;
  id_state_manual?: boolean | null;
  /** Move the glyph to another MOTHRA category (resets its neume label). */
  category?: GlyphCategory | null;
}

export const updateGlyph = (
  id: string,
  glyphId: string,
  patch: UpdateGlyphArgs,
) => http.post<GlyphDTO>(`/sessions/${id}/glyphs/${glyphId}`, patch);

export const saveSession = (id: string) =>
  http.post<SessionDTO>(`/sessions/${id}/save`);

export const completeSession = (id: string) =>
  postForBlob(`/sessions/${id}/complete`);
