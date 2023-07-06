import { useRef, useEffect, useState, useCallback } from 'react';
import * as cv from '@techstark/opencv-js';
import * as ort from 'onnxruntime-web';
import './Localizer.css'

const HEIGHT = 224, WIDTH = 224;

type Props = {
  onSolve?: (img: cv.Mat) => void;
}

function Localizer({ onSolve }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(document.createElement('canvas'));

  const [session, setSession] = useState<ort.InferenceSession | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null)

  // Stores an inference result
  const [{localization, classification}, setResult] = useState({classification: 0.0, localization: [0.0, 0.0, 0.0, 0.0]});

  const isPresent = 0.9 < classification;
  const [x1, y1, x2, y2] = localization;

  useEffect(() => { 
    ort.InferenceSession.create('./localizer.with_runtime_opt.ort', { executionProviders: ['webgl'] })
      .then((session) => setSession(session))
      .catch((e) => console.error(e));
  }, []);

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
    if (!videoRef.current || !session) return;

    const context = canvasRef.current.getContext('2d', { willReadFrequently: true });
    context?.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);

    const img = cv.imread(canvasRef.current);

    cv.resize(img, img, new cv.Size(WIDTH, HEIGHT));
    cv.cvtColor(img, img, cv.COLOR_RGB2GRAY);
    img.convertTo(img, cv.CV_32F, 1. / 255);

    const input = new ort.Tensor('float32', img.data32F, [1, 1, HEIGHT, WIDTH]);
    const { classification, localization } = await session.run({ input });
    const [x1, y1, x2, y2] = localization.data as Float32Array;

    img.delete();

    setResult({
      classification: classification.data[0] as number,
      localization: [x1, y1, x2, y2]
    });
    setTimeout(processFrame, 20);
  }, [session]);

  useEffect(() => {
    if (!videoRef.current || !session || !stream) return;

    videoRef.current.srcObject = stream;
    videoRef.current.play();

    const { width, height } = stream.getVideoTracks()[0].getSettings();
    canvasRef.current.width = width ?? 600;
    canvasRef.current.height = height ?? 600;

    // Everything has loaded start processing frames
    processFrame();

    // TODO: Stop processing frames if component unmounts
    // Unregister webcam if component unmounts
    return () => {
      stream.getTracks()[0].stop();
    };
  }, [processFrame, session, stream]);

  if (!stream) {
    return <div className='loader'>
      <p>You have to give camera permissions to use this webapp.</p>
    </div>;
  }

  if (!session) {
    return <div className='loader'>
      <p>Loading localizer model for sudoku detection.</p>
    </div>;
  }

  const scaleX = (videoRef.current?.clientWidth ?? WIDTH) / WIDTH,
    scaleY = (videoRef.current?.clientHeight ?? HEIGHT) / HEIGHT;

  return (
    <div className='localizer'>
      <div className='camera'>
        <video autoPlay playsInline ref={videoRef} onPlay={processFrame}></video>
        <div className='outline' style={{
          top: y1 * scaleY,
          left: x1 * scaleX,
          width: (x2 - x1) * scaleX,
          height: (y2 - y1) * scaleY,
          opacity: isPresent ? 0.6 : 0
        }}></div>
      </div>
      <button
        disabled={!isPresent}
        onClick={() => {
          // Crop the sudoku out of the image
          const img = cv.imread(canvasRef.current);

          const scaleX = img.cols / WIDTH, scaleY = img.rows / HEIGHT;
          const sudokuRect = new cv.Rect(
            x1 * scaleX, y1 * scaleY,
            (x2 - x1) * scaleX, (y2 - y1) * scaleY
          );
          const crop = img.roi(sudokuRect);

          onSolve && onSolve(crop);

          crop.delete();
          img.delete();
        }}
        className='solve'>SOLVE</button>
    </div>
  );
}

export default Localizer;
