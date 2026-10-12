/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev: Vite on 5173 proxies /api to the FastAPI server (python -m backend.api, port 8000).
// Prod: `npm run build` -> dist/, served by FastAPI itself at / (single port).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
  test: { environment: "node", include: ["src/**/*.test.ts"] },
});
