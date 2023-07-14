import { useContext, useEffect, useState } from 'react';
import * as ort from 'onnxruntime-web';
import * as cv from '@techstark/opencv-js';
import { ORTContext } from '../App';
import { DIGIT_IMG_HEIGHT, DIGIT_IMG_WIDTH } from '../constants';

const IMG_SIZE = DIGIT_IMG_HEIGHT * DIGIT_IMG_WIDTH;


const B = 3; // Border
const W = DIGIT_IMG_WIDTH + 2 * B, H = DIGIT_IMG_HEIGHT + 2 * B;

function threshold(img: cv.Mat) {
    const thresh = new cv.Mat(img.rows, img.cols, cv.CV_8UC1);
    cv.cvtColor(img, thresh, cv.COLOR_RGB2GRAY);
    cv.GaussianBlur(thresh, thresh, new cv.Size(3, 3), 3)
    cv.adaptiveThreshold(thresh, thresh, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 5, 3);
    return thresh;
}

function removeLines(thresh: cv.Mat) {
    const lines = new cv.Mat();

    const M = cv.Mat.ones(3, 3, cv.CV_8U);
    const dilated = new cv.Mat(thresh.rows, thresh.cols, cv.CV_8U);
    cv.dilate(thresh, dilated, M);

    const minLength = Math.min(thresh.rows, thresh.cols)/9 * 1.2;
    const threshold = minLength * 2;

    cv.HoughLinesP(dilated, lines, 2, Math.PI / 180, threshold, minLength, 1);
    for (let i = 0; i < lines.rows; ++i) {
      const startPoint = new cv.Point(lines.data32S[i * 4], lines.data32S[i * 4 + 1]);
      const endPoint = new cv.Point(lines.data32S[i * 4 + 2], lines.data32S[i * 4 + 3]);
      cv.line(thresh, startPoint, endPoint, [0, 0, 0, 255], 5);
    }

    dilated.delete(), M.delete(), lines.delete();
}

function warpPerspective(thresh: cv.Mat) {
    const topL = [thresh.cols, thresh.rows], topR = [0, thresh.rows],
      bottomL = [thresh.cols, 0], bottomR = [0, 0];
    for (let i = 0; i < thresh.data.length; i++) {
      if (thresh.data[i] < 1) continue;

      const y = Math.floor(i / thresh.cols), x = i % thresh.cols;
      if (x + y < topL[0] + topL[1])
        topL[0] = x, topL[1] = y;
      else if (topR[0] - topR[1] < x-y)
        topR[0] = x, topR[1] = y;
      else if (x - y < bottomL[0] - bottomL[1])
        bottomL[0] = x, bottomL[1] = y;
      else if (bottomR[0]+bottomR[1]<x+y)
        bottomR[0] = x, bottomR[1] = y;
    }
    const src = cv.matFromArray(4, 1, cv.CV_32FC2, [...topL, ...topR, ...bottomL, ...bottomR]);
    const dst = cv.matFromArray(4, 1, cv.CV_32FC2, [0,0, 9*W-1,0, 0,9*H-1, 9*W-1,9*H-1]);
    const M = cv.getPerspectiveTransform(src, dst);
    src.delete(), dst.delete();

    const warped = new cv.Mat(H, W, thresh.type());
    cv.warpPerspective(thresh, warped, M, new cv.Size(9*W, 9*H));
    M.delete();

    return warped;
}

export default function useDigitRecognition(img: cv.Mat) {
  const [digits, setDigits] = useState<{digit: number, prob: number}[]>([]);
  const { classifier } = useContext(ORTContext);

  useEffect(() => {
    const thresh = threshold(img);
    const warped = warpPerspective(thresh);
    removeLines(warped);

    // Collect data into one buffer so we can pass it to a tensor
    const buffer = new Float32Array(9 * 9 * IMG_SIZE);
    for (let row = 0; row < 9; ++row) {
      for (let col = 0; col < 9; ++col) {
        const rect = new cv.Rect(col * W + B, row * W + B, DIGIT_IMG_WIDTH, DIGIT_IMG_HEIGHT);
        const region = warped.roi(rect);
        region.convertTo(region, cv.CV_32F, 1. / 255);

        buffer.set(region.data32F, (row * 9 + col) * IMG_SIZE);

        region.delete();
      }
    }
    const input = new ort.Tensor('float32', buffer, [9 * 9, 1, DIGIT_IMG_HEIGHT, DIGIT_IMG_WIDTH]);

    classifier.run({ input })
      .then(({ classification }) => {
        const digits = [];

        const [rows, cols] = classification.dims;
        const data = classification.data as unknown as number[];

        for (let row = 0; row < rows; ++row) {
          let maxIdx = 0, maxProb = 0;
          for (let col = 0; col < cols; ++col) {
            if (maxProb < data[row * cols + col]) {
              maxIdx = col;
              maxProb = data[row * cols + col];
            }
          }
          digits.push({ digit: maxIdx, prob: maxProb });
        }
        setDigits(digits);
      });

    thresh.delete(), warped.delete();
  }, [img, classifier])

  return digits;
}
