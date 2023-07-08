
import * as ort from 'onnxruntime-web';
import * as cv from '@techstark/opencv-js';
import { useContext, useEffect, useMemo, useState } from 'react';
import { ORTContext } from './App';

type Props = {
  sudokuImg: cv.Mat;
};

const IN_SIZE = 28;

function preprocess(img: cv.Mat) {
  // Resize img to appropriate size
  cv.resize(img, img, new cv.Size(9 * IN_SIZE, 9 * IN_SIZE));

  // Convert to grayscale
  cv.cvtColor(img, img, cv.COLOR_RGB2GRAY);

  // Perform thresholding, so the lines are easier to detect
  cv.adaptiveThreshold(img, img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 15);

  // Create dilated copy of the img where the lines have more thickness
  const dilated = new cv.Mat();
  const M = cv.Mat.ones(3, 3, cv.CV_8U);
  cv.dilate(img, dilated, M);

  // Detect lines
  const lines = new cv.Mat();
  cv.HoughLinesP(dilated, lines, 1, Math.PI / 180.0, 80, 70, 4);

  // Erase lines from image by painting them over with black
  for (let i = 0; i < lines.rows; ++i) {
    const startPoint = new cv.Point(lines.data32S[i * 4], lines.data32S[i * 4 + 1]);
    const endPoint = new cv.Point(lines.data32S[i * 4 + 2], lines.data32S[i * 4 + 3]);
    cv.line(img, startPoint, endPoint, [0, 0, 0, 255], 3);
  }

  // Delete used resources
  M.delete();
  dilated.delete();
}

function Solver({ sudokuImg }: Props) {
  const ortContext = useContext(ORTContext);

  const [sudoku, setSudoku] = useState<number[][] | null>(null);

  useEffect(() => {
    const img = sudokuImg.clone();
    preprocess(img);

    // Split image into 9*9 IN_SIZExIN_SIZE squares
    const buffer = new Float32Array(9 * 9 * IN_SIZE * IN_SIZE);
    for (let row = 0; row < 9; ++row) {
      for (let col = 0; col< 9; ++col) {
        const rect = new cv.Rect(col * IN_SIZE, row * IN_SIZE, IN_SIZE, IN_SIZE);
        const region = img.roi(rect);
        cv.imshow
        region.convertTo(region, cv.CV_32F, 1. / 255);

        buffer.set(region.data32F, (row * 9 + col) * IN_SIZE * IN_SIZE);

        cv.imshow(`output${row}${col}`, region)
        region.delete();
      }
    }
    img.delete();

    const input = new ort.Tensor('float32', buffer, [9 * 9, 1, IN_SIZE, IN_SIZE]);

    ortContext.classifier.run({ input: input })
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
  }, [ortContext.classifier, sudokuImg]); 

  return <div>
    {Array(9).fill(0).map((_, i) =>
      <div key={i}>
        {Array(9).fill(0).map((_, j) =>
          <canvas key={j} style={{ margin: 5 }} id={`output${i}${j}`}></canvas>
        )}
      </div>
    )}
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
