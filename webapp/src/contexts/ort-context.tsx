import { createContext, useContext, useEffect, useState } from 'react';
import * as ort from 'onnxruntime-web';

type ORTContextType = {
  localizer: ort.InferenceSession,
  classifier: ort.InferenceSession
}

export const ORTContext = createContext<ORTContextType>({
  localizer: null as any,
  classifier: null as any,
});

export function useOrtContext() {
  return useContext(ORTContext);
}

export function ORTContextProvider({ children }: { children: any }) {
  const [ortContext, setOrtContext] = useState<{
    localizer?: ort.InferenceSession,
    classifier?: ort.InferenceSession
  }>({});

  useEffect(() => {
    const url = window.location.href;
    ort.InferenceSession.create(`${url}/localizer.with_runtime_opt.ort`, { executionProviders: ['webgl', 'cpu'] })
      .then((session) => setOrtContext((ctx) => ({...ctx, localizer: session})))
      .catch((e) => console.error(e));

    ort.InferenceSession.create(`${url}/digitclassifier.with_runtime_opt.ort`, { executionProviders: ['webgl', 'cpu'] })
     .then((session) => setOrtContext((ctx) => ({...ctx, classifier: session})))
     .catch((e) => console.error(e));
  }, []);

  if (!ortContext.classifier || !ortContext.localizer) {
    return <div className='h-screen flex justify-center items-center'>
      <svg className="w-6 mr-4 animate-spin text-blue-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
      </svg>
      <p className='text-2xl text-blue-600'>
        Loading models...
      </p>
    </div>;
  }
  return <ORTContext.Provider value={ortContext as any}>
    {children}
  </ORTContext.Provider>;
}
