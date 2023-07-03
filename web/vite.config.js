import { defineConfig } from 'vite'
import { viteStaticCopy } from 'vite-plugin-static-copy';
import mkcert from 'vite-plugin-mkcert';

// https://vitejs.dev/config/
export default defineConfig({
  server: { https: true },
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
  ],
})
