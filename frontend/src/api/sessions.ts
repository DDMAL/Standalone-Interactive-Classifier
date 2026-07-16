import { http, postForBlob } from "@/api/client";
import type { ClassifierBackend } from "@/store/uiStore";
import type {
  AnnotationFormat,
  BinarizationMethod,
  GlyphCategory,
  GlyphDTO,
  SessionDTO,
  SessionSummary,
  TrainingPresetDTO,
} from "@/types/api";

export interface CreateSessionArgs {
  pageImage: File;
  annotations: File;
  annotationsFormat: AnnotationFormat;
  classNames?: string[];
  /** Uploaded GameraXML (.xml) training-set files; glyphs are concatenated. */
  trainingFiles?: File[];
  /** Precomputed SSL embeddings (.npz) for a trainingFiles entry, matched by
   *  filename stem — see POST /sessions. Unlocks the ssl_fusion backend for
   *  that upload without a live crop. */
  trainingEmbeddings?: File[];
  /** Source page image(s) a trainingFiles entry's glyphs were cropped from,
   *  matched by bbox + mask — an alternative to trainingEmbeddings. */
  trainingImages?: File[];
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
  for (const file of args.trainingEmbeddings ?? []) {
    form.append("training_embeddings", file);
  }
  for (const file of args.trainingImages ?? []) {
    form.append("training_images", file);
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
  /** Precomputed SSL embeddings (.npz) for a trainingFiles entry, matched by
   *  filename stem — see {@link CreateSessionArgs.trainingEmbeddings}. */
  trainingEmbeddings?: File[];
  /** Source page image(s) for a trainingFiles entry's glyphs, matched by
   *  bbox + mask — see {@link CreateSessionArgs.trainingImages}. */
  trainingImages?: File[];
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
  for (const file of args.trainingEmbeddings ?? []) {
    form.append("training_embeddings", file);
  }
  for (const file of args.trainingImages ?? []) {
    form.append("training_images", file);
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
  http.get<TrainingPresetDTO[]>("/training-presets");

/** List the vocabulary CSV filenames under core/data/train. */
export const listVocabularies = () => http.get<string[]>("/vocabularies");

/** Fetch the distinct class names of a vocabulary CSV for preview. */
export const getVocabularyClasses = (name: string) =>
  http.get<string[]>(`/vocabularies/${encodeURIComponent(name)}/classes`);

/** List stored sessions as lightweight summaries, most-recent first. */
export const listSessions = () => http.get<SessionSummary[]>("/sessions");

export const getSession = (id: string) =>
  http.get<SessionDTO>(`/sessions/${id}`);

export const deleteSession = (id: string) => http.delete(`/sessions/${id}`);

/** Discard every stored session; resolves to the number removed. */
export const clearSessions = () =>
  http.deleteFor<{ deleted: number }>("/sessions");

export const classify = (
  id: string,
  k = 3,
  backend: ClassifierBackend = "knn",
) => http.post<SessionDTO>(`/sessions/${id}/classify`, { k, backend });

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

/** Remove one glyph from the training pool; returns the updated session. */
export const deleteTrainingGlyph = (id: string, glyphId: string) =>
  http.deleteFor<SessionDTO>(`/sessions/${id}/training-glyphs/${glyphId}`);

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

/**
 * Which sections to fold into the exported GameraXML. Mirrors the boolean
 * flags on POST /sessions/{id}/complete; at least one must be true.
 */
export interface ExportSelection {
  /** Every working glyph on the annotated page. */
  page?: boolean;
  /** Only the working neumes the user labelled by hand. */
  manualNeumes?: boolean;
  /** Training glyphs that came from a built-in preset. */
  presetTraining?: boolean;
  /** Training glyphs the user uploaded. */
  uploadedTraining?: boolean;
}

export const completeSession = (id: string, selection: ExportSelection) => {
  const params = new URLSearchParams();
  if (selection.page) params.set("page", "true");
  if (selection.manualNeumes) params.set("manual_neumes", "true");
  if (selection.presetTraining) params.set("preset_training", "true");
  if (selection.uploadedTraining) params.set("uploaded_training", "true");
  return postForBlob(`/sessions/${id}/complete?${params.toString()}`);
};

/**
 * Download a companion SSL embeddings (.npz) file for the same section
 * selection as {@link completeSession} -- the ssl_fusion backend's
 * counterpart, letting this exact training set be reused later without a
 * live crop or model pass (see POST /sessions' training_embeddings).
 * Unlike completeSession, this doesn't move the session to EXPORT.
 */
export const exportSessionEmbeddings = (
  id: string,
  selection: ExportSelection,
) => {
  const params = new URLSearchParams();
  if (selection.page) params.set("page", "true");
  if (selection.manualNeumes) params.set("manual_neumes", "true");
  if (selection.presetTraining) params.set("preset_training", "true");
  if (selection.uploadedTraining) params.set("uploaded_training", "true");
  return postForBlob(`/sessions/${id}/export-embeddings?${params.toString()}`);
};
