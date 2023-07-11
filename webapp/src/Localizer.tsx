import { useRef, useEffect, useState } from 'react';
import * as cv from '@techstark/opencv-js';
import { SUDOKU_IMG_HEIGHT, SUDOKU_IMG_WIDTH } from './constants';
import useSudokuLocalizer from './hooks/useSudokuLocalizer';

type Props = {
  onSolve?: (img: cv.Mat) => void;
}

export default function Localizer({ onSolve }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);

  // Stores an inference result
  const {prob, location, cropSudoku} = useSudokuLocalizer(videoRef.current);
  const isPresent = 0.9 < prob;
  const [x1, y1, x2, y2] = location;

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

  useEffect(() => {
    if (!videoRef.current || !stream) return;

    // When stream is available start it in the video element
    videoRef.current.srcObject = stream;

    // Unregister webcam if component unmounts
    return () => {
      stream.getTracks()[0].stop();
    };
  }, [stream]);

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
    <div className='flex justify-between items-center flex-col p-4 sm:p-8'>
      <div className='w-full flex justify-center relative my-4'>
        <video
          className='w-full rounded-xl shadow-xl border-2 border-gray-200'
          autoPlay playsInline ref={videoRef}>
        </video>
        <div className='absolute bg-white transition-all' style={{
          top: y1 * scaleY,
          left: x1 * scaleX,
          width: (x2 - x1) * scaleX,
          height: (y2 - y1) * scaleY,
          opacity: isPresent ? 0.6 : 0
        }}></div>
      </div>

      <button
        disabled={!isPresent}
        className={`transition-all duration-200 mt-4 focus:outline-none focus:ring-4 focus:ring-blue-300 px-8 
                    py-4 font-bold text-sm bg-blue-500 disabled:bg-blue-400 text-white rounded-full shadow-xl 
                    ${isPresent ? '-translate-y-1 opacity-1' : 'opacity-0'}`}
        onClick={() => {
          const crop = cropSudoku();
          onSolve && onSolve(crop);
          // crop.delete();
        }}
      >
        SOLVE
      </button>
    </div>
  );
}
