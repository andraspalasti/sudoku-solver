import * as cv from '@techstark/opencv-js';
import useDigitRecognition from './hooks/useDigitRecognition';
import { useEffect, useState } from 'react';


type Props = {
  sudokuImg: cv.Mat;
  onBack?: () => void;
  onSolve?: (puzzle: number[]) => void;
};

export default function DigitRecognition({ sudokuImg, onBack, onSolve }: Props) {
  const digits = useDigitRecognition(sudokuImg);

  const [sudoku, setSudoku] = useState<{digit: number, certain: boolean}[] | null>(null);
  useEffect(() => {
    setSudoku(digits.map(({ digit, prob }) => ({ digit, certain: 0.95 < prob })));
  }, [digits])

  useEffect(() => {
    setSudoku(null);
  }, [sudokuImg])

  const doesCollide = (row: number, col: number) => {
    if (!sudoku) return;

    const digit = sudoku[row * 9 + col].digit;
    if (digit === 0) return false;

    for (let i = 0; i < 9; ++i) {
      // Check horizontal row
      if (col !== i && sudoku[row * 9 + i].digit === digit) return true;

      // Check vertical column
      if (row !== i && sudoku[i * 9 + col].digit === digit) return true;
    }

    // Check in 3x3 square
    for (let i = Math.floor(row / 3) * 3; i < 3; i++) {
      for (let j = Math.floor(col / 3) * 3; j < 3; j++) {
        if (i !== row && j !== col && sudoku[i * 9 + j].digit === digit) return true;
      }
    }
    return false;
  };

  if (!sudoku) {
    return <div className='h-screen flex justify-center items-center p-4 sm:p-8'>
      <p className='text-xl text-center font-normal text-gray-600'>
        Running recognission model on image.
      </p>
    </div>;
  }

  const hasCollision = sudoku.some((_, i) => doesCollide(Math.floor(i / 9), i % 9));

  return <div className='flex flex-col justify-center items-center p-4 sm:p-8'>
    <p className='m-4 text-xl text-center font-normal text-gray-500'>
      Is this your sudoku? If not edit the digits accordingly.
    </p>
    <div className='m-4 grid grid-cols-9 place-content-center border-2 border-black'>
      {sudoku.map(({ digit, certain }, i) => {
        const row = Math.floor(i / 9), col = i % 9;
        return (
          <div
            key={i}
            className={`aspect-square border border-black rounded-0 
            ${doesCollide(row, col) ? 'bg-red-400' : (certain ? '' : 'bg-yellow-400')}
            ${row % 3 == 0 ? 'border-t-2' : ''} ${col % 3 == 0 ? 'border-l-2' : ''}
            ${row % 3 == 2 ? 'border-b-2' : ''} ${col % 3 == 2 ? 'border-r-2' : ''}`}
          >
            <input
              type='text'
              inputMode='numeric'
              value={digit !== 0 ? digit : ''}
              className={`w-full h-full text-center focus:outline-none 
              ${doesCollide(row, col) ? 'bg-red-400' : (certain ? '' : 'bg-yellow-400')}`}
              onChange={(e: any) => {
                const n = parseInt(e.target.value);
                const d = (Number.isNaN(n) ? 0 : n) % 10;
                setSudoku(sudoku.map((square, j) => i === j ? ({ digit: d, certain: true }) : square));
              }}
              onFocus={(e) => e.target.select()}
            />
          </div>
        );
      })}
    </div>
    <div className='flex'>
      <button
        onClick={onBack}
        className={`mt-4 focus:outline-none focus:ring-4 focus:ring-blue-300 px-8 
                    py-4 font-bold text-sm bg-blue-500 text-white rounded-l-full shadow-xl`}
      >
        BACK
      </button>
      <button
        onClick={() => onSolve && onSolve(sudoku.map(({digit}) => digit))}
        disabled={hasCollision}
        className={`transition-all duration-200 mt-4 focus:outline-none focus:ring-4 focus:ring-blue-300 px-8 
                    py-4 font-bold text-sm bg-blue-500 disabled:bg-blue-400 text-white rounded-r-full shadow-xl 
                    ${!hasCollision ? 'opacity-1' : 'opacity-60'}`}
        >
        SOLVE
      </button>
    </div>
  </div>;
}
