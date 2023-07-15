import { useMemo } from "react";

type Props = {
  puzzle: number[];
  onBack?: () => void;
  onAgain?: () => void;
};

function doesCollide(sudoku: number[][], row: number, col: number) {
  const n = sudoku[row][col];
  if (n === 0) return false;

  for (let i = 0; i < 9; ++i) {
    if (i !== row && sudoku[i][col] === n)
      return true;

    if (i !== col && sudoku[row][i] === n)
      return true;
  }

  const squareRow = Math.floor(row / 3) * 3, squareCol = Math.floor(col / 3) * 3;
  for (let i = squareRow; i < squareRow+3; ++i) {
    for (let j = squareCol; j < squareCol+3; ++j) {
      if (i === row && j === col)
        continue;

      if (sudoku[i][j] === n)
        return true;
    }
  }

  return false;
}

function solve(sudoku: number[][]): number[][] | undefined {
  for (let row = 0; row < 9; ++row) {
    for (let col = 0; col < 9; ++col) {
      if (sudoku[row][col] !== 0)
        continue;

      for (let n = 1; n <= 9; ++n) {
        sudoku[row][col] = n;
        if (doesCollide(sudoku, row, col)) {
          continue;
        }
          
        const solution = solve(sudoku);
        if (solution)
          return solution;
      }
      sudoku[row][col] = 0;
      return undefined;
    }
  }
  return sudoku;
}

export default function Solver({puzzle, onAgain, onBack }: Props) {
  const solution = useMemo(() => {
    const copy = Array(9).fill(0)
      .map((_, i) => puzzle.slice(i * 9, (i + 1) * 9));
    return solve(copy)?.flat();
  }, [puzzle]);
  console.log(solution);

  if (!solution) {
    return <div className='h-screen flex flex-col justify-center items-center p-4 sm:p-8'>
      <p className='text-xl text-center font-normal text-gray-600'>
        The sudoku does not have a solution, are you sure that the sudoku was correctly scanned.
      </p>
      <button
        onClick={onBack}
        className={`transition-all duration-200 mt-8 focus:outline-none focus:ring-4 focus:ring-blue-300 px-8 
                      py-4 font-bold text-sm bg-blue-500 text-white rounded-full shadow-xl`}
      >
        BACK
      </button>
    </div>;
  }

  return <div className='flex flex-col justify-center items-center p-4 sm:p-8'>
    <p className='m-4 text-xl text-center font-normal text-gray-500'>
      The solution of the puzzle:
    </p>
    <div className='m-4 grid grid-cols-9 place-content-center border-2 border-black w-full shadow-lg'>
      {solution.map((digit, i) => {
        const row = Math.floor(i / 9), col = i % 9;
        const isFixed = puzzle[i] !== 0;

        return (
          <div
            key={i}
            className={`aspect-square border border-black rounded-0 flex justify-center items-center
            ${row % 3 == 0 ? 'border-t-2' : ''} ${col % 3 == 0 ? 'border-l-2' : ''}
            ${row % 3 == 2 ? 'border-b-2' : ''} ${col % 3 == 2 ? 'border-r-2' : ''}
            ${isFixed ? 'text-black font-bold' : 'text-gray-500 font-semibold'}`}
          >
            <p>{digit}</p>
          </div>
        )
      })}
    </div>
    <button
      onClick={onAgain}
      className={`transition-all duration-200 mt-8 focus:outline-none focus:ring-4 focus:ring-blue-300 px-8 
                    py-4 font-bold text-sm bg-blue-500 text-white rounded-full shadow-xl`}
    >
      AGAIN
    </button>
  </div>;
}
