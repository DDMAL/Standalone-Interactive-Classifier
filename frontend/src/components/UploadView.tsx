import { Button } from "@/components/ui/Button";
import { useCreateSession } from "@/hooks/useCreateSession";
import { useTrainingSets } from "@/hooks/useTrainingSets";
import { useVocabularies, useVocabularyClasses } from "@/hooks/useVocabularies";
import type { AnnotationFormat } from "@/types/api";
import { type FormEvent, useState } from "react";

export function UploadView() {
  const [pageImage, setPageImage] = useState<File | null>(null);
  const [annotations, setAnnotations] = useState<File | null>(null);
  const [format, setFormat] = useState<AnnotationFormat>("json");
  const [trainingXml, setTrainingXml] = useState("");
  const [vocabulary, setVocabulary] = useState("");
  const create = useCreateSession();
  const trainingSets = useTrainingSets();
  const vocabularies = useVocabularies();
  const vocabClasses = useVocabularyClasses(vocabulary);

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!pageImage || !annotations) return;
    create.mutate({
      pageImage,
      annotations,
      annotationsFormat: format,
      trainingXml: trainingXml || undefined,
      vocabulary: vocabulary || undefined,
    });
  }

  return (
    <div className="flex h-full items-center justify-center bg-slate-50">
      <form
        onSubmit={handleSubmit}
        className="w-[28rem] space-y-4 rounded-lg border border-slate-200 bg-white p-6 shadow-sm"
      >
        <h1 className="text-lg font-semibold text-slate-800">
          Interactive Classifier
        </h1>

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

        <label className="block text-sm">
          <span className="mb-1 block font-medium text-slate-700">
            Training set{" "}
            <span className="font-normal text-slate-400">(optional)</span>
          </span>
          <select
            value={trainingXml}
            onChange={(e) => setTrainingXml(e.target.value)}
            disabled={trainingSets.isLoading}
            className="w-full rounded border border-slate-300 px-2 py-1.5 text-sm"
          >
            <option value="">None</option>
            {(trainingSets.data ?? []).map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
          <span className="mt-1 block text-xs font-normal text-slate-400">
            {trainingSets.isError
              ? "Could not load training sets."
              : trainingXml
                ? "Glyphs will be classified with this training set on start."
                : "Pick a pre-built training set to auto-classify the page."}
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

        {create.isError && (
          <p className="text-sm text-red-600">
            {(create.error as Error).message}
          </p>
        )}

        <Button
          type="submit"
          disabled={!pageImage || !annotations || create.isPending}
          className="w-full"
        >
          {create.isPending ? "Uploading…" : "Start session"}
        </Button>
      </form>
    </div>
  );
}
