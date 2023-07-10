
import * as ort from 'onnxruntime-web';
import * as cv from '@techstark/opencv-js';
import { useContext, useEffect, useMemo, useState } from 'react';
import { ORTContext } from './App';
import { DIGIT_IMG_HEIGHT, DIGIT_IMG_WIDTH } from './constants';
import { preForDigitRec } from './image-processing';


const IMG_SIZE = DIGIT_IMG_HEIGHT * DIGIT_IMG_WIDTH;

type Props = {
  sudokuImg: cv.Mat;
};

function Solver({ sudokuImg }: Props) {
  const ortContext = useContext(ORTContext);

  const [sudoku, setSudoku] = useState<number[][] | null>(null);

  const tensor = useMemo(() => {
    const img = sudokuImg.clone();
    preForDigitRec(img);

    // Split image into 9*9 IN_SIZExIN_SIZE squares
    const buffer = new Float32Array(9 * 9 * IMG_SIZE);
    for (let row = 0; row < 9; ++row) {
      for (let col = 0; col < 9; ++col) {
        const rect = new cv.Rect(col * DIGIT_IMG_WIDTH, row * DIGIT_IMG_HEIGHT, DIGIT_IMG_WIDTH, DIGIT_IMG_HEIGHT);
        const region = img.roi(rect);
        region.convertTo(region, cv.CV_32F, 1. / 255);

        buffer.set(region.data32F, (row * 9 + col) * IMG_SIZE);

        region.delete();
      }
    }
    img.delete();
    return new ort.Tensor('float32', buffer, [9 * 9, 1, DIGIT_IMG_HEIGHT, DIGIT_IMG_WIDTH]);
  }, [sudokuImg]);

  useEffect(() => {
    ortContext.classifier.run({ input: tensor })
      .then(({ classification }) => {
        const sudoku = Array(9).fill(0).map(() => Array(9).fill(0));
        const [rows, cols] = classification.dims

        for (let row = 0; row < rows; ++row) {
          let maxIdx = 0;
          for (let col = 0; col < cols; ++col) {
            if (classification.data[row * cols + maxIdx] < classification.data[row * cols + col]) {
              maxIdx = col;
            }
          }
          sudoku[Math.floor(row / 9)][row % 9] = maxIdx;
        }
        setSudoku(sudoku);
      })
      .catch((e) => console.error(e));
  }, [ortContext.classifier, tensor])

  return <div>
    {sudoku && sudoku.map((row, i) => {
      return <div key={i}>
        {row.map((n, j) => {
          return <span style={{margin: 5}} key={j}>{n}</span>
        })}
      </div>;
    })}
  </div>;
}

export default Solver;
