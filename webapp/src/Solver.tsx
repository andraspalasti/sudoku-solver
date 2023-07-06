
import * as cv from '@techstark/opencv-js';
import { useEffect } from 'react';

type Props = {
  sudoku: cv.Mat;
};

const SIZE = 28 * 9;
function preprocess(img: cv.Mat) {
  // Resize img to appropriate size
  cv.resize(img, img, new cv.Size(SIZE, SIZE));

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

function Solver({ sudoku }: Props) {

  useEffect(() => {
    const img = sudoku.clone();
    preprocess(img);

    cv.imshow('output', img);
    img.delete();
  }, [sudoku]); 

  return <div>
    <canvas id='output'></canvas>
  </div>;
}

export default Solver;
