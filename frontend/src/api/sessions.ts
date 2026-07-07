import { http, postForBlob } from "@/api/client";
import type {
  AnnotationFormat,
  BinarizationMethod,
  GlyphCategory,
  GlyphDTO,
  SessionDTO,
} from "@/types/api";

export interface CreateSessionArgs {
  pageImage: File;
  annotations: File;
  annotationsFormat: AnnotationFormat;
  classNames?: string[];
  /** Uploaded GameraXML (.xml) training-set files; glyphs are concatenated. */
  trainingFiles?: File[];
  /** Built-in preset filenames (see {@link listTrainingPresets}); concatenated ahead of uploads. */
  trainingPresets?: string[];
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
  for (const file of args.trainingFiles ?? []) {
    form.append("training_files", file);
  }
  if (args.trainingPresets && args.trainingPresets.length > 0) {
    form.append("training_presets", JSON.stringify(args.trainingPresets));
  }
  if (args.vocabulary) {
    form.append("vocabulary", args.vocabulary);
  }
  return http.postForm<SessionDTO>("/sessions", form);
}

/** Metadata for a page + bboxes staged by an embedding host (see /staging). */
export interface StagingInfo {
  staging_id: string;
  page_name: string;
  annotations_format: AnnotationFormat;
  annotation_count: number;
}

export const getStaging = (id: string) =>
  http.get<StagingInfo>(`/staging/${id}`);

export interface CreateSessionFromStagingArgs {
  stagingId: string;
  classNames?: string[];
  /** Uploaded GameraXML (.xml) training-set files; glyphs are concatenated. */
  trainingFiles?: File[];
  /** Built-in preset filenames (see {@link listTrainingPresets}); concatenated ahead of uploads. */
  trainingPresets?: string[];
  /** Filename of a vocabulary CSV (see {@link listVocabularies}). */
  vocabulary?: string;
}

/** Create a session from a staged page + bboxes plus the user's choices. */
export function createSessionFromStaging(
  args: CreateSessionFromStagingArgs,
): Promise<SessionDTO> {
  const form = new FormData();
  form.append("staging_id", args.stagingId);
  if (args.classNames && args.classNames.length > 0) {
    form.append("class_names", JSON.stringify(args.classNames));
  }
  for (const file of args.trainingFiles ?? []) {
    form.append("training_files", file);
  }
  if (args.trainingPresets && args.trainingPresets.length > 0) {
    form.append("training_presets", JSON.stringify(args.trainingPresets));
  }
  if (args.vocabulary) {
    form.append("vocabulary", args.vocabulary);
  }
  return http.postForm<SessionDTO>("/sessions/from-staging", form);
}

/** List the built-in training-set preset filenames under core/data/presets. */
export const listTrainingPresets = () =>
  http.get<string[]>("/training-presets");

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

/**
 * Re-binarise the page with a different method and rebuild every glyph
 * mask. Manual labels are kept (matched by glyph id); manual groups/splits
 * reset to the detector's base glyphs. Auto labels stay until the next
 * classify round, which re-derives them from the new masks.
 */
export const rebinarize = (id: string, method: BinarizationMethod) =>
  http.post<SessionDTO>(`/sessions/${id}/binarization`, { method });

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

export const completeSession = (id: string, includeTraining = false) =>
  postForBlob(
    `/sessions/${id}/complete${includeTraining ? "?include_training=true" : ""}`,
  );
