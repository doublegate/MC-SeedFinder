import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  build: {
    // Three.js core is a known large vendor dependency for the 3D map. Keep
    // the warning threshold just above that split chunk so future app-code
    // growth still shows up instead of being hidden by a blanket high limit.
    chunkSizeWarningLimit: 750,
    rolldownOptions: {
      output: {
        codeSplitting: {
          groups: [
            { name: "vendor-react", test: /[\\/]node_modules[\\/](react|react-dom)[\\/]/ },
            { name: "vendor-tauri", test: /[\\/]node_modules[\\/]@tauri-apps[\\/]/ },
            { name: "vendor-r3f", test: /[\\/]node_modules[\\/]@react-three[\\/]/ },
            { name: "vendor-three", test: /[\\/]node_modules[\\/]three[\\/]/ },
          ],
        },
      },
    },
  },
});
