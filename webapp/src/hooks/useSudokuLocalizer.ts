import { useCallback, useContext, useEffect, useRef, useState } from "react";
import * as ort from 'onnxruntime-web';
import * as cv from '@techstark/opencv-js';
import { SUDOKU_IMG_HEIGHT, SUDOKU_IMG_WIDTH } from '../constants';
import { ORTContext } from "../App";

type LocalizerResult = {
  prob: number,
  location: [number, number, number, number],
}

const FPS = 5;

export default function useSudokuLocalizer(video: HTMLVideoElement | null) {
  const timeoutId = useRef<NodeJS.Timeout | null>(null);
  const canvasRef = useRef(document.createElement('canvas'));

  const { localizer } = useContext(ORTContext);

  const [result, setResult] = useState<LocalizerResult>({ location: [0, 0, 0, 0], prob: 0.0 });

  const resources = useRef({
    img: new cv.Mat(),
    resized: new cv.Mat(),
    gray: new cv.Mat(),
    converted: new cv.Mat(),
  });

  const processFrame = useCallback(async () => {
    if (!video) return;
    const { img, resized, gray, converted } = resources.current;

    const ctx = canvasRef.current.getContext('2d', { willReadFrequently: true });
    ctx?.drawImage(video, 0, 0, canvasRef.current.width, canvasRef.current.height);
    img.data.set(ctx?.getImageData(0, 0, canvasRef.current.width, canvasRef.current.height).data);

    cv.resize(img, resized, new cv.Size(SUDOKU_IMG_WIDTH, SUDOKU_IMG_HEIGHT));
    cv.cvtColor(resized, gray, cv.COLOR_RGBA2GRAY);
    gray.convertTo(converted, cv.CV_32F, 1.0 / 255);

    const input = new ort.Tensor('float32', converted.data32F, [1, 1, SUDOKU_IMG_HEIGHT, SUDOKU_IMG_WIDTH]);
    const { classification, localization } = await localizer.run({ input });

    const [x1, y1, x2, y2] = localization.data as Float32Array;
    setResult({
      location: [x1, y1, x2, y2],
      prob: classification.data[0] as number,
    });
  }, [localizer, video]);

  const cropSudoku = useCallback((padding = 0) => {
    const [x1, y1, x2, y2] = result.location;
    const { img } = resources.current;
    const scaleX = img.cols / SUDOKU_IMG_WIDTH, scaleY = img.rows / SUDOKU_IMG_HEIGHT;
    return img.roi(new cv.Rect(
      x1 * scaleX - padding, y1 * scaleY - padding,
      (x2 - x1) * scaleX + 2 * padding, (y2 - y1) * scaleY + 2 * padding
    ));
  }, [result.location])

  useEffect(() => {

    function onPlay() {
      if (!video) return;
      if (!(video.srcObject instanceof MediaStream)) return;

      const track = video.srcObject.getVideoTracks()[0];
      const { width, height } = track.getSettings();

      canvasRef.current.width = width!;
      canvasRef.current.height = height!;
      resources.current = {
        img: new cv.Mat(height!, width!, cv.CV_8UC4),
        resized: new cv.Mat(SUDOKU_IMG_HEIGHT, SUDOKU_IMG_WIDTH, cv.CV_8UC4),
        gray: new cv.Mat(SUDOKU_IMG_HEIGHT, SUDOKU_IMG_WIDTH, cv.CV_8UC1),
        converted: new cv.Mat(SUDOKU_IMG_HEIGHT, SUDOKU_IMG_WIDTH, cv.CV_32F),
      };

      const process = () => {
        const start = performance.now();
        processFrame();
        const delay = 1000 / FPS - (performance.now() - start);
        timeoutId.current = setTimeout(process, delay);
      }
      process();
    }

    video?.addEventListener('play', onPlay);

    return () => {
      video?.removeEventListener('play', onPlay)

      const { img, resized, gray, converted } = resources.current;
      if (!img.isDeleted()) {
        img.delete();
        resized.delete();
        gray.delete();
        converted.delete();
      }

      if (timeoutId.current) clearTimeout(timeoutId.current);
    };
  }, [processFrame, video]);

  return {
    ...result,
    cropSudoku,
  }
}