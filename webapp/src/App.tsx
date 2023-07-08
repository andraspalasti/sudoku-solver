import { createContext, useEffect, useState } from 'react';
import * as cv from '@techstark/opencv-js';
import * as ort from 'onnxruntime-web';
import Localizer from './Localizer';
import Solver from './Solver';

type ORTContextType = {
  localizer: ort.InferenceSession,
  classifier: ort.InferenceSession
}

export const ORTContext = createContext<ORTContextType>({
  localizer: null as any,
  classifier: null as any,
});

function App() {
  const [ortContext, setOrtContext] = useState<{
    localizer?: ort.InferenceSession,
    classifier?: ort.InferenceSession
  }>({});
  const [sudoku, setSudoku] = useState<cv.Mat | null>(null);

  useEffect(() => {
    ort.InferenceSession.create('./localizer.with_runtime_opt.ort', { executionProviders: ['webgl'] })
      .then((session) => setOrtContext((ctx) => ({...ctx, localizer: session})))
      .catch((e) => console.error(e));

    ort.InferenceSession.create('./digitclassifier.with_runtime_opt.ort', { executionProviders: ['webgl'] })
      .then((session) => setOrtContext((ctx) => ({...ctx, classifier: session})))
      .catch((e) => console.error(e));
  }, []);

  if (!ortContext.classifier || !ortContext.localizer) {
    return <div>
      <p>Loading models.</p>
    </div>;
  }

  return <ORTContext.Provider value={ortContext as any}>
    {sudoku ?
      <Solver sudokuImg={sudoku} /> :
      <Localizer
        onSolve={(img) => {
          const newSudoku = new cv.Mat();
          img.copyTo(newSudoku);
          setSudoku(newSudoku);
        }}
      />}
  </ORTContext.Provider>;
}

export default App;
