import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      keyframes: {
        // Gentle "breathing" pulse for the amber hover highlight on SVG
        // bounding boxes (opacity works on both fill and stroke).
        "bbox-pulse": {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.45" },
        },
        // Amber glow ring that pulses on the hovered glyph tile without
        // affecting its layout or the glyph image inside it.
        "amber-glow": {
          "0%, 100%": { boxShadow: "0 0 0 0 rgba(245, 158, 11, 0)" },
          "50%": { boxShadow: "0 0 0 4px rgba(245, 158, 11, 0.35)" },
        },
      },
      animation: {
        "bbox-pulse": "bbox-pulse 1.1s ease-in-out infinite",
        "amber-glow": "amber-glow 1.1s ease-in-out infinite",
      },
    },
  },
  plugins: [],
} satisfies Config;
