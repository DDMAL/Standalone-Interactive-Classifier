import { getVocabularyClasses, listVocabularies } from "@/api/sessions";
import { useQuery } from "@tanstack/react-query";

/** Fetch the vocabulary CSV filenames for the upload dropdown. */
export function useVocabularies() {
  return useQuery({
    queryKey: ["vocabularies"],
    queryFn: listVocabularies,
    staleTime: Number.POSITIVE_INFINITY,
  });
}

/** Fetch the class names of a vocabulary CSV; disabled until one is picked. */
export function useVocabularyClasses(name: string) {
  return useQuery({
    queryKey: ["vocabulary-classes", name],
    queryFn: () => getVocabularyClasses(name),
    enabled: name !== "",
    staleTime: Number.POSITIVE_INFINITY,
  });
}
