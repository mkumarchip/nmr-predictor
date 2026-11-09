// Source placeholder for the intended Vite/React frontend.
// The deployed offline fallback is frontend/index.html because npm registry
// access was not available in this environment during implementation.
import React from "react";
import { createRoot } from "react-dom/client";

function App() {
  return (
    <main>
      <h1>NMR Predictor v2</h1>
      <p>Run the FastAPI-served offline fallback or install npm dependencies to wire nmrium here.</p>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
