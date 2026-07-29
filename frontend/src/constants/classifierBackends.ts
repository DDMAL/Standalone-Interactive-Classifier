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
    label: "Pre-trained + SVM",
    title:
      "Optional: self-supervised (SSL) features fused with handcrafted " +
      "features, classified with a linear-kernel SVM. Requires the server " +
      "to have the ssl extra installed and a fine-tuned checkpoint " +
      "configured. Training data must carry precomputed SSL embeddings or " +
      "a real-pixel crop — presets marked SSL-compatible qualify " +
      "automatically, and uploaded GameraXML files qualify when paired " +
      "with a companion embeddings file or their original source image(s).",
  },
];
