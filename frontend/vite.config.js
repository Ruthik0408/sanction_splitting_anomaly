import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "/ui/",
  plugins: [react()],
  server: {
    proxy: {
      "/check_purchase": "http://localhost:5000",
      "/existing_purchases": "http://localhost:5000",
      "/health": "http://localhost:5000",
      "/ready": "http://localhost:5000",
    },
  },
});
