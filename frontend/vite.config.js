import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The dev server proxies /chat, /health, /ingest to the FastAPI backend so
// the browser sees same-origin requests during development (no CORS pain).
// VITE_API_URL (src/api.js) is used for production builds instead.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/chat': { target: 'http://localhost:8000', changeOrigin: true },
      '/health': { target: 'http://localhost:8000', changeOrigin: true },
      '/ingest': { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
})
