import { useState } from 'react';
import * as cv from '@techstark/opencv-js';
import Localizer from './Localizer';
import Solver from './Solver';

function App() {
  const [sudoku, setSudoku] = useState<cv.Mat | null>(null);

  if (!sudoku) {
    return <Localizer
      onSolve={(img) => {
        const newSudoku = new cv.Mat();
        img.copyTo(newSudoku);
        setSudoku(newSudoku);
      }}
    />;
  }

  return <Solver sudoku={sudoku} />;
}

export default App;
