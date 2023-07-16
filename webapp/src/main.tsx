import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.tsx';
import './index.css';

// import * as ort from 'onnxruntime-web';
// ort.env.wasm.wasmPaths = {
// 'ort-wasm-simd.wasm': 'sudoku-solver/ort-wasm-simd.wasm',
// };

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
