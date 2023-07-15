import { useReducer } from 'react';
import * as cv from '@techstark/opencv-js';
import { ORTContextProvider } from './contexts/ort-context';
import Localizer from './Localizer';
import DigitRecognition from './DigitRecognition';
import Solver from './Solver';

type ReducerState =
  | { type: 'scanning' }
  | { type: 'recognising', img: cv.Mat }
  | { type: 'solving', img: cv.Mat, puzzle: number[] };

type ReducerAction =
  | { type: 'scan', img: cv.Mat }
  | { type: 'solve', puzzle: number[] }
  | { type: 'back' }
  | { type: 'again' };

function reducer(state: ReducerState, action: ReducerAction): ReducerState {
  if (state.type === 'scanning' && action.type === 'scan')
    return { type: 'recognising', img: action.img };

  if (state.type === 'recognising') {
    if (action.type === 'back')
      return { type: 'scanning' };

    if (action.type === 'solve')
      return { type: 'solving', puzzle: action.puzzle, img: state.img };
  }

  if (state.type === 'solving') {
    if (action.type === 'back')
      return { type: 'recognising', img: state.img };

    if (action.type === 'again')
      return { type: 'scanning' };
  }

  throw Error('Unknown action.');
}

export default function App() {
  const [state, dispatch] = useReducer(reducer, { type: 'scanning' });

  return <ORTContextProvider>
    <canvas id='out'></canvas>
    {(() => {
      switch (state.type) {
        case 'scanning':
          return <Localizer
            onScan={(img) => {
              const copy = new cv.Mat();
              img.copyTo(copy);
              dispatch({ type: 'scan', img: copy });
            }}
          />;
        case 'recognising':
          return <DigitRecognition
            sudokuImg={state.img}
            onBack={() => dispatch({ type: 'back' })}
            onSolve={(puzzle) => dispatch({ type: 'solve', puzzle })}
          />;
        case 'solving':
          return <Solver
            puzzle={state.puzzle}
            onBack={() => dispatch({ type: 'back' })}
            onAgain={() => dispatch({type: 'again'})}
          />;
      }
    })()}
  </ORTContextProvider>;
}
