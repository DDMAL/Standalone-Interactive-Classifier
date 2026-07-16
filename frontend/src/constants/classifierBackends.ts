import type { ClassifierBackend } from "@/store/uiStore";

/** Shared between UploadView (where the model is chosen) and Toolbar
 *  (which just displays the active one for the rest of the session). */
export const CLASSIFIER_BACKENDS: {
  value: ClassifierBackend;
  label: string;
  title: string;
}[] = [
  {
    value: "knn",
    label: "HC + kNN",
    title: "Default: handcrafted-feature k-nearest-neighbours classifier.",
  },
  {
    value: "ssl_fusion",
    label: "Pre-trained + LR",
    title:
      "Optional: self-supervised (SSL) features fused with handcrafted " +
      "features, classified with logistic regression. Requires the server " +
      "to have the ssl extra installed and a fine-tuned checkpoint " +
      "configured. Training data must carry precomputed SSL embeddings — " +
      "only presets marked accordingly qualify; uploaded GameraXML files " +
      "and incompatible presets can't be used with this model.",
  },
];
