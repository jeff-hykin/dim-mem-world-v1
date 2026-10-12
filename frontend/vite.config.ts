import process from "node:process"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// served at /apps/<name>/, so every asset URL is relative; the dev server forwards this app's API to its server through
// a running Desktop (DESKTOP_URL, default 7077) and Desktop's /recordings to Desktop. DESKTOP_URL=http://127.0.0.1:7099
// with `node dev/mock_server.mjs` running answers both from a dev-only mock.
const desktop = process.env.DESKTOP_URL ?? "http://127.0.0.1:7077"
const app = process.env.APP_NAME ?? "dim-mem-world"

export default defineConfig({
    base: "./",
    plugins: [react()],
    server: {
        proxy: {
            "/api": { target: `${desktop}/apps/${app}`, changeOrigin: true },
            "/recordings": { target: desktop, changeOrigin: true },
        },
    },
    build: { target: "es2022", chunkSizeWarningLimit: 1500 },
})
