import { useRef, useEffect, useState, useCallback, useContext } from 'react';
import * as cv from '@techstark/opencv-js';
import * as ort from 'onnxruntime-web';
import { ORTContext } from './App';
import { SUDOKU_IMG_HEIGHT, SUDOKU_IMG_WIDTH } from './constants';

const FPS = 5;

type Props = {
  onSolve?: (img: cv.Mat) => void;
}

function Localizer({ onSolve }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(document.createElement('canvas'));
  const ortContext = useContext(ORTContext);

  // Stores an inference result
  const [{localization, classification}, setResult] = useState({classification: 0.0, localization: [0.0, 0.0, 0.0, 0.0]});
  const isPresent = 0.9 < classification;
  const [x1, y1, x2, y2] = localization;

  const [stream, setStream] = useState<MediaStream | null>(null);
  useEffect(() => { 
    navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        aspectRatio: { exact: 1.0 },
        facingMode: { ideal: 'environment' },
      }
    })
      .then((stream) => setStream(stream))
      .catch((e) => console.error(e));
  }, []);

  const processFrame = useCallback(async () => {
    if (!videoRef.current) return;

    const start = performance.now();
    const context = canvasRef.current.getContext('2d', { willReadFrequently: true });
    context?.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);
    const img = cv.imread(canvasRef.current);

    cv.resize(img, img, new cv.Size(SUDOKU_IMG_WIDTH, SUDOKU_IMG_WIDTH));
    cv.cvtColor(img, img, cv.COLOR_RGB2GRAY);
    img.convertTo(img, cv.CV_32F, 1. / 255);

    const input = new ort.Tensor('float32', img.data32F, [1, 1, SUDOKU_IMG_HEIGHT, SUDOKU_IMG_WIDTH]);
    const { classification, localization } = await ortContext.localizer.run({ input });
    const [x1, y1, x2, y2] = localization.data as Float32Array;

    img.delete();

    setResult({
      classification: classification.data[0] as number,
      localization: [x1, y1, x2, y2]
    });
    const end = performance.now();
    const freq = Math.max(5, 1000 / FPS - (end - start)) - 4;
    setTimeout(processFrame, freq);
  }, [ortContext.localizer]);

  useEffect(() => {
    if (!videoRef.current || !stream) return;

    // When stream is available start it in the video element
    videoRef.current.srcObject = stream;
    // videoRef.current.play();

    const { width, height } = stream.getVideoTracks()[0].getSettings();
    canvasRef.current.width = width ?? 600;
    canvasRef.current.height = height ?? 600;

    // Everything has loaded start processing frames
    processFrame();

    // Unregister webcam if component unmounts
    return () => {
      stream.getTracks()[0].stop();
    };
  }, [processFrame, stream]);

  if (!stream) {
    return <div className='h-screen flex justify-center items-center p-4 sm:p-8'>
      <p className='text-xl text-center font-normal text-gray-600'>
        You have to give camera permissions to use this webapp.
      </p>
    </div>;
  }

  const scaleX = (videoRef.current?.clientWidth ?? 0) / SUDOKU_IMG_WIDTH,
    scaleY = (videoRef.current?.clientHeight ?? 0) / SUDOKU_IMG_HEIGHT;

  return (
    <div className='h-screen flex justify-center items-center flex-col p-4 sm:p-8'>
      <canvas id='out'></canvas>
      <div className='w-full flex justify-center relative my-4'>
        <video
          className='w-full rounded-xl shadow-xl border-2 border-gray-200'
          autoPlay playsInline ref={videoRef}>
        </video>
        <div className='absolute bg-white transition-all duration-300' style={{
          top: y1 * scaleY,
          left: x1 * scaleX,
          width: (x2 - x1) * scaleX,
          height: (y2 - y1) * scaleY,
          opacity: isPresent ? 0.6 : 0
        }}></div>
      </div>

      <button
        disabled={!isPresent}
        className={`transition-all duration-200 mt-4 px-8 py-4 font-semibold text-sm bg-blue-500 text-white rounded-full shadow-xl 
                    ${isPresent ? 'translate-y-1 opacity-1' : 'opacity-0'}`}
        onClick={() => {
          // Crop the sudoku out of the image
          const img = cv.imread(canvasRef.current);

          const scaleX = img.cols / SUDOKU_IMG_WIDTH, scaleY = img.rows / SUDOKU_IMG_HEIGHT;
          const sudokuRect = new cv.Rect(
            x1 * scaleX, y1 * scaleY,
            (x2 - x1) * scaleX, (y2 - y1) * scaleY
          );
          const crop = img.roi(sudokuRect);

          onSolve && onSolve(crop);

          crop.delete();
          img.delete();
        }}
      >
        SOLVE
      </button>

      {/* Filler to push elements to top */}
      <div className='w-full h-1/6'></div>
    </div>
  );
}

export default Localizer;
