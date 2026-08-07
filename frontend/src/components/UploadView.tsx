import { getStaging } from "@/api/sessions";
import { SessionResumeList } from "@/components/SessionResumeList";
import { Button } from "@/components/ui/Button";
import { useAutoExport } from "@/hooks/useAutoExport";
import {
  useCreateSession,
  useCreateSessionFromStaging,
} from "@/hooks/useCreateSession";
import { useTrainingPresets } from "@/hooks/useTrainingPresets";
import { useVocabularies, useVocabularyClasses } from "@/hooks/useVocabularies";
import type { AnnotationFormat } from "@/types/api";
import { useQuery } from "@tanstack/react-query";
import { type FormEvent, useEffect, useState } from "react";

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

  // Presets are mutually exclusive: checking one unchecks the rest. The state
  // stays an array because the API takes a list (`training_presets`) and an
  // uploaded set can still be combined with a preset — only preset-vs-preset
  // is exclusive.
  function togglePreset(name: string, checked: boolean) {
    setSelectedPresets(checked ? [name] : []);
  }

  const totalTrainingSets = selectedPresets.length + trainingFiles.length;

  const staging = useQuery({
    queryKey: ["staging", stagedId],
    queryFn: () => getStaging(stagedId as string),
    enabled: !!stagedId,
    staleTime: Number.POSITIVE_INFINITY,
  });

  const active = stagedId ? createFromStaging : create;

  // Embedded in a host (mothra): the host may have picked a training set at the
  // batch level. Announce readiness, then adopt whatever it pushes back so the
  // user doesn't re-select the same training set on every page — but never
  // clobber a selection already made here.
  useEffect(() => {
    if (window.parent === window) return; // standalone — nothing to sync
    function onMessage(e: MessageEvent) {
      if (e.source !== window.parent) return;
      const data = e.data;
      if (data?.type !== "ic:prefill-training") return;
      if (Array.isArray(data.presets))
        setSelectedPresets((prev) => (prev.length ? prev : data.presets));
      if (Array.isArray(data.files))
        setTrainingFiles((prev) => (prev.length ? prev : data.files));
    }
    window.addEventListener("message", onMessage);
    window.parent.postMessage({ type: "ic:ready" }, "*");
    return () => window.removeEventListener("message", onMessage);
  }, []);

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

  // One-click shortcut: create the session and run a classify round over the
  // page — no session view. Embedded in a host (mothra), it hands the page to
  // the host's encode queue; standalone, it downloads the GameraXML. Shares
  // the same inputs as start.
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

  // Embedded in a host (mothra) via iframe → "queue page" semantics; standalone
  // → "auto-export" (download). Drives the button's wording only.
  const embedded = window.parent !== window;
  const anyPending = active.isPending || autoExport.isPending;
  const inputsMissing = stagedId ? !staging.data : !pageImage || !annotations;
  const submitDisabled = anyPending || inputsMissing;
  // Queueing/auto-export always runs a classify round, which needs a non-empty
  // training pool — so grey it out until the user picks a preset or uploads a set.
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
                  {(presets.data ?? []).map((name) => (
                    <label key={name} className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={selectedPresets.includes(name)}
                        onChange={(e) => togglePreset(name, e.target.checked)}
                      />
                      <span className="text-slate-700">{name}</span>
                    </label>
                  ))}
                </div>
              )}
            </div>

            <label className="block">
              <span className="mb-1 block text-xs font-medium text-slate-600">
                Upload
              </span>
              <input
                type="file"
                accept=".xml"
                multiple
                onChange={(e) =>
                  setTrainingFiles(Array.from(e.target.files ?? []))
                }
                className="block w-full text-sm"
              />
            </label>

            {trainingFiles.length > 0 && (
              <ul className="space-y-0.5 text-xs text-slate-500">
                {trainingFiles.map((f, i) => (
                  <li key={`${f.name}-${i}`} className="truncate">
                    {f.name}
                  </li>
                ))}
              </ul>
            )}

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
          {autoExport.isSuccess && !embedded && (
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
                  ? embedded
                    ? "Add a preset or upload a training set to enable queuing"
                    : "Add a preset or upload a training set to enable auto-export"
                  : embedded
                    ? "Classify the page with the training set and add it straight to the encode queue"
                    : "Create the session, run one classification round, and download the page as GameraXML"
              }
            >
              {autoExport.isPending
                ? embedded
                  ? "Queuing…"
                  : "Auto-exporting…"
                : embedded
                  ? "Queue page"
                  : "Auto-export"}
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
