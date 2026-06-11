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
  /** Filename of a vocabulary CSV (see {@link listVocabularies}). */
  vocabulary?: string;
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
  if (args.vocabulary) {
    form.append("vocabulary", args.vocabulary);
  }
  return http.postForm<SessionDTO>("/sessions", form);
}

/** List the pre-built training-set XML filenames under core/data/derived. */
export const listTrainingSets = () => http.get<string[]>("/training-sets");

/** List the vocabulary CSV filenames under core/data/train. */
export const listVocabularies = () => http.get<string[]>("/vocabularies");

/** Fetch the distinct class names of a vocabulary CSV for preview. */
export const getVocabularyClasses = (name: string) =>
  http.get<string[]>(`/vocabularies/${encodeURIComponent(name)}/classes`);

export const getSession = (id: string) =>
  http.get<SessionDTO>(`/sessions/${id}`);

export const deleteSession = (id: string) => http.delete(`/sessions/${id}`);

export const classify = (id: string, k = 3) =>
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

export const deleteGlyph = (id: string, glyphId: string) =>
  http.delete(`/sessions/${id}/glyphs/${glyphId}`);

export interface ManualGroupArgs {
  glyph_ids: string[];
  class_name: string;
}

export const manualGroup = (id: string, body: ManualGroupArgs) =>
  http.post<GlyphDTO>(`/sessions/${id}/group`, body);

export interface SplitArgs {
  /** Page-coordinate rectangles as [ulx, uly, ncols, nrows] tuples. */
  regions: [number, number, number, number][];
}

export const splitGlyph = (id: string, glyphId: string, body: SplitArgs) =>
  http.post<GlyphDTO[]>(`/sessions/${id}/glyphs/${glyphId}/split`, body);

export const renameClass = (id: string, name: string, new_name: string) =>
  http.post<SessionDTO>(
    `/sessions/${id}/classes/${encodeURIComponent(name)}/rename`,
    { new_name },
  );

export const deleteClass = (id: string, name: string) =>
  http.deleteFor<SessionDTO>(
    `/sessions/${id}/classes/${encodeURIComponent(name)}`,
  );

export const saveSession = (id: string) =>
  http.post<SessionDTO>(`/sessions/${id}/save`);

export const completeSession = (id: string) =>
  postForBlob(`/sessions/${id}/complete`);
