import { getStaging } from "@/api/sessions";
import { SessionResumeList } from "@/components/SessionResumeList";
import { Button } from "@/components/ui/Button";
import { CLASSIFIER_BACKENDS } from "@/constants/classifierBackends";
import { useAutoExport } from "@/hooks/useAutoExport";
import {
  useCreateSession,
  useCreateSessionFromStaging,
} from "@/hooks/useCreateSession";
import { useTrainingPresets } from "@/hooks/useTrainingPresets";
import { useVocabularies, useVocabularyClasses } from "@/hooks/useVocabularies";
import { useUiStore } from "@/store/uiStore";
import type { AnnotationFormat } from "@/types/api";
import { useQuery } from "@tanstack/react-query";
import { clsx } from "clsx";
import { type FormEvent, useState } from "react";

interface UploadViewProps {
  // When set, the page image + bboxes have been staged by an embedding host
  // (mothra). The page/annotation inputs are replaced by a read-only summary
  // and the session is created via /sessions/from-staging — the user only
  // supplies training data + vocabulary here.
  stagedId?: string;
}

export function UploadView({ stagedId }: UploadViewProps = {}) {
  const [pageImage, setPageImage] = useState<File | null>(null);
  const [annotations, setAnnotations] = useState<File | null>(null);
  const [format, setFormat] = useState<AnnotationFormat>("json");
  const [trainingFiles, setTrainingFiles] = useState<File[]>([]);
  const [selectedPresets, setSelectedPresets] = useState<string[]>([]);
  const [vocabulary, setVocabulary] = useState("");
  const create = useCreateSession();
  const createFromStaging = useCreateSessionFromStaging();
  const autoExport = useAutoExport();
  const presets = useTrainingPresets();
  const vocabularies = useVocabularies();
  const vocabClasses = useVocabularyClasses(vocabulary);
  const classifierBackend = useUiStore((s) => s.classifierBackend);
  const setClassifierBackend = useUiStore((s) => s.setClassifierBackend);

  // ssl_fusion needs precomputed SSL embeddings, which only some presets
  // ship and no uploaded GameraXML file ever carries (see
  // ic_core.ssl_preset_embeddings) — so switching to it drops any
  // now-unusable training data rather than leaving a silently-ignored
  // selection sitting in the form.
  function handleBackendChange(backend: "knn" | "ssl_fusion") {
    if (backend === classifierBackend) return;
    setClassifierBackend(backend);
    if (backend === "ssl_fusion") {
      const compatibleNames = new Set(
        (presets.data ?? []).filter((p) => p.ssl_compatible).map((p) => p.name),
      );
      setSelectedPresets((prev) => prev.filter((n) => compatibleNames.has(n)));
      setTrainingFiles([]);
    }
  }

  const sslFusionSelected = classifierBackend === "ssl_fusion";

  function togglePreset(name: string, checked: boolean) {
    setSelectedPresets((prev) =>
      checked ? [...prev, name] : prev.filter((n) => n !== name),
    );
  }

  const totalTrainingSets = selectedPresets.length + trainingFiles.length;

  const staging = useQuery({
    queryKey: ["staging", stagedId],
    queryFn: () => getStaging(stagedId as string),
    enabled: !!stagedId,
    staleTime: Number.POSITIVE_INFINITY,
  });

  const active = stagedId ? createFromStaging : create;

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const training = trainingFiles.length > 0 ? trainingFiles : undefined;
    const presetNames =
      selectedPresets.length > 0 ? selectedPresets : undefined;
    if (stagedId) {
      createFromStaging.mutate({
        stagingId: stagedId,
        trainingFiles: training,
        trainingPresets: presetNames,
        vocabulary: vocabulary || undefined,
      });
      return;
    }
    if (!pageImage || !annotations) return;
    create.mutate({
      pageImage,
      annotations,
      annotationsFormat: format,
      trainingFiles: training,
      trainingPresets: presetNames,
      vocabulary: vocabulary || undefined,
    });
  }

  // One-click shortcut: create the session, run a classify round, and download
  // the page as GameraXML — no session view. Shares the same inputs as start.
  function handleAutoExport() {
    const training = trainingFiles.length > 0 ? trainingFiles : undefined;
    const presetNames =
      selectedPresets.length > 0 ? selectedPresets : undefined;
    if (stagedId) {
      autoExport.mutate({
        kind: "staging",
        args: {
          stagingId: stagedId,
          trainingFiles: training,
          trainingPresets: presetNames,
          vocabulary: vocabulary || undefined,
        },
      });
      return;
    }
    if (!pageImage || !annotations) return;
    autoExport.mutate({
      kind: "upload",
      args: {
        pageImage,
        annotations,
        annotationsFormat: format,
        trainingFiles: training,
        trainingPresets: presetNames,
        vocabulary: vocabulary || undefined,
      },
    });
  }

  const anyPending = active.isPending || autoExport.isPending;
  const inputsMissing = stagedId ? !staging.data : !pageImage || !annotations;
  const submitDisabled = anyPending || inputsMissing;
  // Auto-export always runs a classify round, which needs a non-empty training
  // pool — so grey it out until the user picks a preset or uploads a set.
  const autoExportDisabled = submitDisabled || totalTrainingSets === 0;

  return (
    <div className="flex h-full bg-slate-50">
      {/* Standalone-only: mothra resumes via a deep-link, not this list. */}
      {!stagedId && <SessionResumeList />}
      <div className="flex flex-1 flex-col items-center justify-center gap-4 overflow-y-auto p-4">
        <form
          onSubmit={handleSubmit}
          className="w-[28rem] space-y-4 rounded-lg border border-slate-200 bg-white p-6 shadow-sm"
        >
          <h1 className="text-lg font-semibold text-slate-800">
            Interactive Classifier
          </h1>

          {stagedId ? (
            <div className="rounded border border-slate-200 bg-slate-50 p-3 text-sm">
              <span className="mb-1 block font-medium text-slate-700">
                Page &amp; detections
              </span>
              {staging.isLoading ? (
                <span className="text-slate-400">Loading staged page…</span>
              ) : staging.isError ? (
                <span className="text-red-600">
                  Could not load the staged page. The link may have expired.
                </span>
              ) : (
                <span className="text-slate-600">
                  {staging.data?.page_name} — {staging.data?.annotation_count}{" "}
                  detection
                  {staging.data?.annotation_count === 1 ? "" : "s"} (
                  {staging.data?.annotations_format === "yolo"
                    ? "YOLO"
                    : "MOTHRA JSON"}
                  )
                </span>
              )}
              <span className="mt-1 block text-xs text-slate-400">
                Provided by mothra. Add training data and a vocabulary below,
                then start the session.
              </span>
            </div>
          ) : (
            <>
              <label className="block text-sm">
                <span className="mb-1 block font-medium text-slate-700">
                  Page image
                </span>
                <input
                  type="file"
                  accept="image/*"
                  onChange={(e) => setPageImage(e.target.files?.[0] ?? null)}
                  className="block w-full text-sm"
                />
              </label>

              <label className="block text-sm">
                <span className="mb-1 block font-medium text-slate-700">
                  Annotations file
                </span>
                <input
                  type="file"
                  accept=".json,.txt"
                  onChange={(e) => setAnnotations(e.target.files?.[0] ?? null)}
                  className="block w-full text-sm"
                />
              </label>

              <label className="block text-sm">
                <span className="mb-1 block font-medium text-slate-700">
                  Annotation format
                </span>
                <select
                  value={format}
                  onChange={(e) =>
                    setFormat(e.target.value as AnnotationFormat)
                  }
                  className="w-full rounded border border-slate-300 px-2 py-1.5"
                >
                  <option value="json">MOTHRA JSON</option>
                  <option value="yolo">YOLO TXT</option>
                </select>
              </label>
            </>
          )}

          <div className="space-y-2 text-sm">
            <span className="block font-medium text-slate-700">Model</span>
            <div className="flex overflow-hidden rounded border border-slate-300 w-fit">
              {CLASSIFIER_BACKENDS.map(({ value, label, title }) => (
                <button
                  key={value}
                  type="button"
                  onClick={() => handleBackendChange(value)}
                  title={title}
                  className={clsx(
                    "px-3 py-1 text-xs font-medium transition-colors",
                    value === classifierBackend
                      ? "bg-blue-600 text-white"
                      : "bg-white text-slate-700 hover:bg-slate-100",
                  )}
                >
                  {label}
                </button>
              ))}
            </div>
            {sslFusionSelected && (
              <span className="block text-xs font-normal text-slate-400">
                Only presets with precomputed SSL embeddings can be used as
                training data with this model — uploaded GameraXML files don't
                qualify.
              </span>
            )}
          </div>

          <div className="space-y-2 text-sm">
            <span className="block font-medium text-slate-700">
              Training data{" "}
              <span className="font-normal text-slate-400">(optional)</span>
            </span>

            <div>
              <span className="mb-1 block text-xs font-medium text-slate-600">
                Presets
              </span>
              {presets.isLoading ? (
                <span className="text-xs text-slate-400">Loading presets…</span>
              ) : presets.isError ? (
                <span className="text-xs text-red-600">
                  Could not load presets.
                </span>
              ) : (presets.data ?? []).length === 0 ? (
                <span className="text-xs text-slate-400">
                  No presets available.
                </span>
              ) : (
                <div className="space-y-1">
                  {(presets.data ?? []).map(({ name, ssl_compatible }) => {
                    const disabled = sslFusionSelected && !ssl_compatible;
                    return (
                      <label
                        key={name}
                        className={clsx(
                          "flex items-center gap-2",
                          disabled && "cursor-not-allowed opacity-50",
                        )}
                        title={
                          disabled
                            ? "No precomputed SSL embeddings — can't be used with the Pre-trained + LR model."
                            : undefined
                        }
                      >
                        <input
                          type="checkbox"
                          checked={selectedPresets.includes(name)}
                          disabled={disabled}
                          onChange={(e) => togglePreset(name, e.target.checked)}
                        />
                        <span className="text-slate-700">{name}</span>
                      </label>
                    );
                  })}
                </div>
              )}
            </div>

            <label
              className={clsx(
                "block",
                sslFusionSelected && "cursor-not-allowed opacity-50",
              )}
              title={
                sslFusionSelected
                  ? "Uploaded GameraXML files never carry SSL embeddings — can't be used with the Pre-trained + LR model."
                  : undefined
              }
            >
              <span className="mb-1 block text-xs font-medium text-slate-600">
                Upload
              </span>
              <input
                type="file"
                accept=".xml"
                multiple
                disabled={sslFusionSelected}
                onChange={(e) =>
                  setTrainingFiles(Array.from(e.target.files ?? []))
                }
                className="block w-full text-sm"
              />
            </label>

            <span className="block text-xs font-normal text-slate-400">
              {totalTrainingSets > 0
                ? `${totalTrainingSets} training ${
                    totalTrainingSets === 1 ? "set" : "sets"
                  } will be combined and classified on start.`
                : "Pick presets and/or upload GameraXML (.xml) training sets to auto-classify the page."}
            </span>
          </div>

          <label className="block text-sm">
            <span className="mb-1 block font-medium text-slate-700">
              Vocabulary{" "}
              <span className="font-normal text-slate-400">(optional)</span>
            </span>
            <select
              value={vocabulary}
              onChange={(e) => setVocabulary(e.target.value)}
              disabled={vocabularies.isLoading}
              className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm"
            >
              <option value="">None</option>
              {(vocabularies.data ?? []).map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
            <span className="mt-1 block text-xs font-normal text-slate-400">
              {vocabularies.isError
                ? "Could not load vocabularies."
                : "Pick a vocabulary to seed the available class names."}
            </span>
          </label>

          {vocabulary && (
            <div className="text-sm">
              <span className="mb-1 block font-medium text-slate-700">
                Available classes{" "}
                {vocabClasses.data && (
                  <span className="font-normal text-slate-400">
                    ({vocabClasses.data.length})
                  </span>
                )}
              </span>
              <textarea
                readOnly
                value={
                  vocabClasses.isLoading
                    ? "Loading…"
                    : vocabClasses.isError
                      ? "Could not load class names."
                      : (vocabClasses.data ?? []).join("\n")
                }
                rows={6}
                className="w-full resize-y rounded border border-slate-200 bg-slate-50 px-2 py-1.5 font-mono text-xs text-slate-600"
              />
            </div>
          )}

          {active.isError && (
            <p className="text-sm text-red-600">
              {(active.error as Error).message}
            </p>
          )}
          {autoExport.isError && (
            <p className="text-sm text-red-600">
              {(autoExport.error as Error).message}
            </p>
          )}
          {autoExport.isSuccess && (
            <p className="text-sm text-green-600">
              Exported {autoExport.data}.
            </p>
          )}

          <div className="flex gap-2">
            <Button
              type="button"
              variant="secondary"
              onClick={handleAutoExport}
              disabled={autoExportDisabled}
              className="flex-1"
              title={
                totalTrainingSets === 0
                  ? "Add a preset or upload a training set to enable auto-export"
                  : "Create the session, run one classification round, and download the page as GameraXML"
              }
            >
              {autoExport.isPending ? "Auto-exporting…" : "Auto-export"}
            </Button>
            <Button type="submit" disabled={submitDisabled} className="flex-1">
              {active.isPending ? "Uploading…" : "Start session"}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}
