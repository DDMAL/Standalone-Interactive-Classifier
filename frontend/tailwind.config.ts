import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Mothra brand palette (matches the embedding landing-page host).
        mothra: {
          cyan: "#4AADAA", // primary accent (buttons, selected states)
          "cyan-dark": "#1E6B70", // primary hover / active
          "cyan-faint": "#C8E6E3", // faint cyan (secondary buttons, tinted bg)
          "cyan-muted": "#B0CDC9", // faint-cyan hover
          teal: "#1D3335", // dark teal text on light/faint backgrounds
        },
      },
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
