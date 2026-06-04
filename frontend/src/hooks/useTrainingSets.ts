import { listTrainingSets } from "@/api/sessions";
import { useQuery } from "@tanstack/react-query";

/** Fetch the pre-built training-set filenames for the upload dropdown. */
export function useTrainingSets() {
  return useQuery({
    queryKey: ["training-sets"],
    queryFn: listTrainingSets,
    staleTime: Infinity,
  });
}
