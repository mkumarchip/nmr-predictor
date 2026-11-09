# NMR Predictor v3

Local FastAPI app for predicting 1H and 13C NMR shifts from a confirmed molecule, using isolated subprocess runners for the installed model environments.

The v3 interface adds atom-aligned consensus and disagreement coloring,
experimental Mnova-report/JCAMP/JEOL overlays with model ranking, assignment and image
exports, and a linked simulated 1H-13C HSQC panel. All parsing, comparison, and
export work remains local and offline.

The orchestrator never imports CSP5, CASCADE, NMRNet, or FullSSPrUCe packages. Each model runs through its own runner script and returns one JSON payload on stdout.

The models themselves are **not included**. Install each one separately under its own license. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for links, licenses, and the papers to cite.

## What Is Included

```text
nmr-app/
├── config.example.json
├── server.py
├── structure.py
├── multiplicity.py
├── simulate.py
├── runners/
│   ├── run_csp5.py
│   ├── run_cascade.py
│   ├── run_nmrnet.py
│   └── run_fullsspruce.py
├── frontend/
│   ├── index.html
│   ├── package.json
│   └── src/main.jsx
├── README.md
├── LICENSE
└── THIRD_PARTY_NOTICES.md
```

The UI served by `server.py` is `frontend/index.html`. It is dependency-free at runtime.

## Current Model Status

Copy `config.example.json` to `config.json` (git-ignored) and point each model's `python` and `cwd` at your local environments. Without a `config.json` the server falls back to the example file.

- `csp5`: enabled, shifts for 1H and 13C.
- `cascade`: enabled, shifts and confidence for 13C.
- `fullsspruce`: enabled, shifts and real H-H scalar couplings for 1H multiplets.
- `nmrnet`: enabled after C/H calibration. Calibration is stored as `nmrnet_calibration.json` in the NMRNet working directory.

## Run

```bash
cp config.example.json config.json   # then edit paths
python server.py
```

Open:

```text
http://127.0.0.1:8000
```

For a fresh light backend env:

```bash
pip install fastapi uvicorn rdkit py2opsin --default-timeout=120 --retries 15 --trusted-host pypi.org --trusted-host files.pythonhosted.org
python server.py
```

`py2opsin` is optional at runtime for the included common-name fallbacks, but needed for broader offline name resolution. A standalone `opsin.jar` can also be used by setting `OPSIN_JAR=/path/to/opsin.jar`.

Experimental import additionally uses `jcamp` and the JEOL-capable upstream
`nmrglue` build. The current PyPI `nmrglue` release does not contain its JEOL
reader:

```bash
python -m pip install --default-timeout=120 --retries 15 \
  --trusted-host pypi.org --trusted-host files.pythonhosted.org jcamp
python -m pip install --default-timeout=120 --retries 15 \
  --trusted-host github.com --trusted-host codeload.github.com \
  --no-deps https://github.com/jjhelmus/nmrglue/archive/refs/heads/master.zip
```

## API

```bash
curl http://127.0.0.1:8000/api/models
```

```bash
curl -X POST http://127.0.0.1:8000/api/structure \
  -H 'Content-Type: application/json' \
  -d '{"type":"smiles","value":"CCO"}'
```

```bash
curl -X POST http://127.0.0.1:8000/api/structure \
  -H 'Content-Type: application/json' \
  -d '{"type":"name","value":"aspirin"}'
```

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"smiles":"CCO","models":["csp5","cascade","fullsspruce"]}'
```

If `models` is omitted, `/api/predict` runs every enabled model. A failed model returns its own `error` entry while successful models still return spectra.

Experimental peak reports and text spectra use the same JSON endpoint:

```bash
curl -X POST http://127.0.0.1:8000/api/experimental \
  -H 'Content-Type: application/json' \
  -d '{"nucleus":"AUTO","text":"1H NMR (400 MHz, CDCl3) δ 7.26 (d, J = 7.1 Hz, 2H)"}'
```

The browser sends `.jdf` and `.mnova` uploads to this endpoint as base64 binary
data. The 100 MB request limit is enforced before parsing.

## Experimental Formats

- Mnova Multiplet Analysis text: combined 1H/13C reports, ranges,
  multiplicities, integrations, and multiple J values.
- JCAMP-DX: compressed/full XY spectra and peak tables through `jcamp`, with
  basic peak picking for full spectra.
- JEOL JDF: 1D FID or processed spectra through `nmrglue`; raw FIDs receive
  apodization, zero-fill, FT, attempted auto-phase, normalization, and peak
  picking. The UI labels these results as auto-processed and reviewable.
- Native Mnova projects: embedded JCAMP/report entries are imported when the
  project is a readable ZIP container. Other proprietary projects return export
  instructions for JCAMP-DX or Multiplet Analysis text.

## Viewer

The current UI provides the accepted interim viewer:

- SMILES, molfile/SDF upload, and offline name input
- confirmed RDKit SVG structure display
- model checkboxes
- overlaid simulated 1D spectra
- reversed ppm axes
- wheel zoom, drag pan, click peak readout, shift-drag integration
- multiplet labels and J values

FullSSPrUCe multiplets are drawn from real H-H couplings and are labeled as non-estimated. Shift-only models use the first-order n+1 proton multiplicity heuristic and label those multiplets as estimated.

`frontend/package.json` includes a React/Vite/nmrium scaffold for the target nmrium integration, but npm package installation was not completed in this environment. The served offline UI does not make frontend network requests at runtime.

## Notes

Cold starts are slow because every request launches subprocesses and loads model weights. A persistent worker per model environment is the likely next performance improvement.

## License

Copyright (c) 2026 mkumarchip. All rights reserved. This repository is shared for demonstration only. See [LICENSE](LICENSE). The third-party models it calls keep their own licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
