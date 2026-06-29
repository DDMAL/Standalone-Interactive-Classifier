import { getStaging } from "@/api/sessions";
import { Button } from "@/components/ui/Button";
import {
  useCreateSession,
  useCreateSessionFromStaging,
} from "@/hooks/useCreateSession";
import { useVocabularies, useVocabularyClasses } from "@/hooks/useVocabularies";
import type { AnnotationFormat } from "@/types/api";
import { useQuery } from "@tanstack/react-query";
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
  const [vocabulary, setVocabulary] = useState("");
  const create = useCreateSession();
  const createFromStaging = useCreateSessionFromStaging();
  const vocabularies = useVocabularies();
  const vocabClasses = useVocabularyClasses(vocabulary);

  const staging = useQuery({
    queryKey: ["staging", stagedId],
    queryFn: () => getStaging(stagedId as string),
    enabled: !!stagedId,
    staleTime: Infinity,
  });

  const active = stagedId ? createFromStaging : create;

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (stagedId) {
      createFromStaging.mutate({
        stagingId: stagedId,
        trainingFiles: trainingFiles.length > 0 ? trainingFiles : undefined,
        vocabulary: vocabulary || undefined,
      });
      return;
    }
    if (!pageImage || !annotations) return;
    create.mutate({
      pageImage,
      annotations,
      annotationsFormat: format,
      trainingFiles: trainingFiles.length > 0 ? trainingFiles : undefined,
      vocabulary: vocabulary || undefined,
    });
  }

  const submitDisabled =
    active.isPending ||
    (stagedId ? !staging.data : !pageImage || !annotations);

  return (
    <div className="flex h-full items-center justify-center bg-slate-50">
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
                {staging.data?.page_name} —{" "}
                {staging.data?.annotation_count} detection
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
                onChange={(e) => setFormat(e.target.value as AnnotationFormat)}
                className="w-full rounded border border-slate-300 px-2 py-1.5"
              >
                <option value="json">MOTHRA JSON</option>
                <option value="yolo">YOLO TXT</option>
              </select>
            </label>
          </>
        )}

        <label className="block text-sm">
          <span className="mb-1 block font-medium text-slate-700">
            Training set{" "}
            <span className="font-normal text-slate-400">(optional)</span>
          </span>
          <input
            type="file"
            accept=".xml"
            multiple
            onChange={(e) => setTrainingFiles(Array.from(e.target.files ?? []))}
            className="block w-full text-sm"
          />
          <span className="mt-1 block text-xs font-normal text-slate-400">
            {trainingFiles.length > 0
              ? `${trainingFiles.length} training ${
                  trainingFiles.length === 1 ? "set" : "sets"
                } will be combined and classified on start.`
              : "Upload one or more GameraXML (.xml) training sets to auto-classify the page."}
          </span>
        </label>

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

        <Button type="submit" disabled={submitDisabled} className="w-full">
          {active.isPending ? "Uploading…" : "Start session"}
        </Button>
      </form>
    </div>
  );
}
