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

// Default sample page, bundled under frontend/public/samples (the Hufnagel
// test pair from core/data/test). It ships inside the frontend so it is
// fetchable both from the Vite dev server and the single-origin production
// build, where the API serves the built assets. Pre-loading it lets the user
// start a session in one click; either file can still be replaced.
const SAMPLE_IMAGE_URL = "/samples/image_hfn_sample.png";
const SAMPLE_IMAGE_NAME = "image_hfn_sample.png";
const SAMPLE_ANNOTATIONS_URL = "/samples/image_hfn_sample_annotations.json";
const SAMPLE_ANNOTATIONS_NAME = "image_hfn_sample_annotations.json";
// Preset checked by default on load, when the API offers it.
const DEFAULT_PRESET = "Hufnagel.xml";

/** Fetch a bundled asset and wrap it as a File for the upload form. */
async function fetchAsFile(
  url: string,
  name: string,
  type: string,
): Promise<File> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load ${name}`);
  const blob = await res.blob();
  return new File([blob], name, { type });
}

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
    staleTime: Infinity,
  });

  const active = stagedId ? createFromStaging : create;

  // Pre-load the bundled Hufnagel sample page so the form is ready to submit
  // without picking files. Skipped in the staged (mothra) flow, where the page
  // is supplied upstream. `prev ?? file` avoids clobbering anything the user
  // selected before the async fetch resolved.
  useEffect(() => {
    if (stagedId) return;
    let cancelled = false;
    (async () => {
      try {
        const [img, ann] = await Promise.all([
          fetchAsFile(SAMPLE_IMAGE_URL, SAMPLE_IMAGE_NAME, "image/png"),
          fetchAsFile(
            SAMPLE_ANNOTATIONS_URL,
            SAMPLE_ANNOTATIONS_NAME,
            "application/json",
          ),
        ]);
        if (cancelled) return;
        setPageImage((prev) => prev ?? img);
        setAnnotations((prev) => prev ?? ann);
      } catch {
        // The sample is a convenience default; on failure just leave the
        // inputs empty for the user to fill in.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [stagedId]);

  // Default-select the Hufnagel preset once the list has loaded, if present.
  // Gated on an empty selection so it does not fight a user's own choice, and
  // keyed on the loaded data so unchecking it does not re-add it.
  const presetNames = presets.data;
  useEffect(() => {
    if (!presetNames?.includes(DEFAULT_PRESET)) return;
    setSelectedPresets((prev) => (prev.length === 0 ? [DEFAULT_PRESET] : prev));
  }, [presetNames]);

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
                {pageImage && (
                  <span className="mt-1 block text-xs text-slate-500">
                    Loaded: {pageImage.name}
                  </span>
                )}
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
                {annotations && (
                  <span className="mt-1 block text-xs text-slate-500">
                    Loaded: {annotations.name}
                  </span>
                )}
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
