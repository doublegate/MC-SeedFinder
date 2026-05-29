import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  build: {
    // Three.js core is a known large vendor dependency for the 3D map. Keep
    // the warning threshold just above that split chunk so future app-code
    // growth still shows up instead of being hidden by a blanket high limit.
    chunkSizeWarningLimit: 750,
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes("node_modules")) return undefined;
          if (id.includes("/node_modules/react/") || id.includes("/node_modules/react-dom/")) {
            return "vendor-react";
          }
          if (id.includes("/node_modules/@tauri-apps/")) {
            return "vendor-tauri";
          }
          if (id.includes("/node_modules/@react-three/")) {
            return "vendor-r3f";
          }
          if (id.includes("/node_modules/three/")) {
            return "vendor-three";
          }
          return undefined;
        },
      },
    },
  },
});
