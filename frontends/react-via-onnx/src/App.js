// import Home from "./pages/Home";

import { useEffect, useState } from "react";

export default function App() {
  const [ort, setOrt] = useState(null);

  const loadWasm = async () => {
    try {
      const ort = await import("onnxruntime-web");
      const modelUrl = process.env.PUBLIC_URL + "/model.onnx";
      console.log(modelUrl);
      const newOrtSession = await ort.InferenceSession.create(modelUrl);

      setOrt(ort);
      console.log("ort import ok");
    } catch (err) {
      console.error(`Unexpected error in loadWasm. [Message: ${err.message}]`);
    }
  };

  useEffect(() => {
    loadWasm();
  }, []);

  return (
    <div className="App">
      {/* <Home ort={ort} /> */}
      <h1>Hello World</h1>
    </div>
  );
}
