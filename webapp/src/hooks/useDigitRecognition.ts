import { useCallback, useContext, useEffect, useRef, useState } from 'react';
import * as ort from 'onnxruntime-web';
import * as cv from '@techstark/opencv-js';
import { ORTContext } from '../App';
import { DIGIT_IMG_HEIGHT, DIGIT_IMG_WIDTH } from '../constants';

const IMG_SIZE = DIGIT_IMG_HEIGHT * DIGIT_IMG_WIDTH;

export default function useDigitRecognition(img: cv.Mat) {
  const [digits, setDigits] = useState<number[]>([]);
  const { classifier } = useContext(ORTContext);

  const resources = useRef<{
      gray: cv.Mat,
      thresh: cv.Mat,
      dilated: cv.Mat,
      M: cv.Mat,
  }>();

  const preprocess = useCallback((img: cv.Mat) => {
    const { gray, thresh, dilated, M } = resources.current!;
    cv.cvtColor(img, gray, cv.COLOR_RGB2GRAY);
    cv.adaptiveThreshold(gray, thresh, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 15);
    cv.dilate(thresh, dilated, M);

    const lines = new cv.Mat();
    cv.HoughLinesP(dilated, lines, 1, Math.PI / 180.0, 80, 70, 4);

    for (let i = 0; i < lines.rows; ++i) {
      const startPoint = new cv.Point(lines.data32S[i * 4], lines.data32S[i * 4 + 1]);
      const endPoint = new cv.Point(lines.data32S[i * 4 + 2], lines.data32S[i * 4 + 3]);
      cv.line(thresh, startPoint, endPoint, [0, 0, 0, 255], 3);
    }
    return thresh;
  }, []);

  const boundingRect = useCallback((thresh: cv.Mat) => {
    let minX = thresh.cols, minY = thresh.rows, maxX = 0, maxY = 0;
    for (let i = 0; i < thresh.data.length; ++i) {
      if (0 < thresh.data[i]) {
        const y = Math.floor(i / thresh.cols), x = i % thresh.cols;
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      }
    }
    return [minX, minY, maxX, maxY];
  }, []);

  useEffect(() => {
    if (!resources.current) {
      resources.current = {
        gray: new cv.Mat(),
        thresh: new cv.Mat(),
        dilated: new cv.Mat(),
        M: cv.Mat.ones(3, 3, cv.CV_8U),
      };
    }

    const thresh = preprocess(img);
    const [minX, minY, maxX, maxY] = boundingRect(thresh);

    const crop = thresh.roi(new cv.Rect(minX, minY, maxX - minX, maxY - minY));

    const padding = 5;
    cv.copyMakeBorder(crop, crop, padding, padding, padding, padding, cv.BORDER_CONSTANT, [0, 0, 0, 255]);
    cv.resize(crop, crop, new cv.Size(9 * DIGIT_IMG_WIDTH, 9 * DIGIT_IMG_HEIGHT));

    // Collect data into one buffer so we can pass it to a tensor
    const buffer = new Float32Array(9 * 9 * IMG_SIZE);
    for (let row = 0; row < 9; ++row) {
      for (let col = 0; col < 9; ++col) {
        const rect = new cv.Rect(col * DIGIT_IMG_WIDTH, row * DIGIT_IMG_HEIGHT, DIGIT_IMG_WIDTH, DIGIT_IMG_HEIGHT);
        const region = crop.roi(rect);
        region.convertTo(region, cv.CV_32F, 1. / 255);

        buffer.set(region.data32F, (row * 9 + col) * IMG_SIZE);

        region.delete();
      }
    }
    crop.delete();

    const input = new ort.Tensor('float32', buffer, [9 * 9, 1, DIGIT_IMG_HEIGHT, DIGIT_IMG_WIDTH]);
    classifier.run({ input })
      .then(({ classification }) => {
        const digits = [];
        const [rows, cols] = classification.dims;
        for (let row = 0; row < rows; ++row) {
          let maxIdx = 0;
          for (let col = 0; col < cols; ++col) {
            if (classification.data[row * cols + maxIdx] < classification.data[row * cols + col]) {
              maxIdx = col;
            }
          }
          digits.push(maxIdx);
        }
        setDigits(digits);
      });

    return () => {
      // Free resources on unmount
      if (!resources.current) return;
      const { gray, thresh, dilated, M } = resources.current;
      gray.delete(), thresh.delete(), dilated.delete(), M.delete();
      resources.current = undefined;
    };
  }, [boundingRect, preprocess, img, classifier])

  return digits;
}
