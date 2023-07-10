import * as cv from '@techstark/opencv-js';
import { DIGIT_IMG_HEIGHT, DIGIT_IMG_WIDTH } from './constants.ts';


/**
 * Preprocessing function for digit detection. 
 * 
 * @param img The image of the sudoku. Does not return anything, instead it modifies the img parameter.
 */
export function preForDigitRec(img: cv.Mat) {
  // Resize img to appropriate size
  cv.resize(img, img, new cv.Size(9 * DIGIT_IMG_WIDTH, 9 * DIGIT_IMG_HEIGHT));

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
