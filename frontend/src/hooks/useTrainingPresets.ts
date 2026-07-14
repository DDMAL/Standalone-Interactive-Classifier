import { listTrainingPresets } from "@/api/sessions";
import { useQuery } from "@tanstack/react-query";

/** Fetch the built-in training-set preset filenames for the upload screen. */
export function useTrainingPresets() {
  return useQuery({
    queryKey: ["training-presets"],
    queryFn: listTrainingPresets,
    staleTime: Number.POSITIVE_INFINITY,
  });
}
