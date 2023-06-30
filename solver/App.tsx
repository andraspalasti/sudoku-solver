import { StatusBar } from "expo-status-bar";
import { Button, StyleSheet, Text, View } from "react-native";

import * as ort from "onnxruntime-react-native";
import { Asset } from "expo-asset";
import { useEffect, useState } from "react";

export default function App() {
  const [session, setSession] = useState<ort.InferenceSession | null>(null);

  const infer = async () => {
    if (!session) throw new Error("Session is null cannot do inference like this")

    const inputData = new ort.Tensor(new Float32Array(400 * 400), [1, 400, 400]);
    const start = performance.now();
    const result = await session.run({ 'input': inputData });
    const end = performance.now();
    console.log(result);
    console.log(end - start);
  };

  useEffect(() => {
    (async () => {
      try {
        const assets = await Asset.loadAsync(require("./assets/localizer.ort"));
        const modelUri = assets[0].localUri;
        if (!modelUri) {
          console.error("model uri is null");
          return;
        }
        let session = await ort.InferenceSession.create(modelUri);
        setSession(session);
      } catch (error) {
        console.error(error)
      }
    })()
  }, []);

  if (!session) {
    return (
      <View style={styles.container}>
        <Text>loading model</Text>
        <StatusBar style="auto" />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Text>using ONNX Runtime for React Native</Text>
      <Button title="Infer" onPress={infer}></Button>
      <StatusBar style="auto" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
});
