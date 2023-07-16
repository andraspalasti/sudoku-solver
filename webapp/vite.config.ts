import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import mkcert from 'vite-plugin-mkcert';
import { viteStaticCopy } from 'vite-plugin-static-copy';
import { VitePWA } from 'vite-plugin-pwa';

// https://vitejs.dev/config/
export default defineConfig({
  base: 'sudoku-solver',
  plugins: [
    mkcert(),
    viteStaticCopy({
      targets: [
        {
          src: 'node_modules/onnxruntime-web/dist/*.wasm',
          dest: '.'
        }
      ]
    }),
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['favicon.ico', 'apple-touch-icon.png', 'mask-icon.svg'],
      manifest: {
        name: 'Sudoku Solver',
        short_name: 'Sudoku Solver',
        description: 'This sudoku app lets you scan a sudoku than solve it.',
        theme_color: '#ffffff',
        icons: [
          {
            src: 'pwa-192x192.png',
            sizes: '192x192',
            type: 'image/png',
            purpose: 'any'
          },
          {
           src: 'pwa-512x512.png',
           sizes: '512x512',
           type: 'image/png',
           purpose: 'any'
          },
          {
            src: 'apple-touch-icon.png',
            sizes: '180x180',
            type: 'image/png',
            purpose: 'any'
          },
          {
            src: 'maskable_icon.png',
            sizes: '225x225',
            type: 'image/png',
            purpose: 'maskable'
          }
        ],
        start_url: '/sudoku-solver/',
        background_color: '#e8ebf2',
        display: 'standalone',
        orientation: 'portrait'
      }
    })
  ],
});
