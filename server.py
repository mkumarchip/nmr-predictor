import base64
import binascii
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from consensus import build_consensus, consensus_shifts
from experimental import parse_experimental
from hsqc import build_hsqc
from multiplicity import CARBON_TOLERANCE, PROTON_TOLERANCE, annotate_shifts
from simulate import simulate_payload, simulate_spectrum
from structure import canonicalize_structure


APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / "config.json"
EXAMPLE_CONFIG_PATH = APP_DIR / "config.example.json"
FRONTEND_PATH = APP_DIR / "frontend" / "index.html"
NUCLEUS_TOLERANCE = {"C": CARBON_TOLERANCE, "H": PROTON_TOLERANCE}

app = FastAPI(title="NMR Multi-Model Predictor")


def load_config() -> dict[str, Any]:
    path = CONFIG_PATH if CONFIG_PATH.exists() else EXAMPLE_CONFIG_PATH
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def abs_script_path(script: str) -> str:
    path = Path(script).expanduser()
    if not path.is_absolute():
        path = APP_DIR / path
    return str(path.resolve())


def model_public_entry(key: str, cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": key,
        "label": cfg.get("label", key),
        "nuclei": cfg.get("nuclei", []),
        "color": cfg.get("color", "#ffffff"),
        "enabled": bool(cfg.get("enabled", False)),
        "couplings": bool(cfg.get("couplings", False)),
    }


def parse_runner_json(stdout: str) -> dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("runner produced no stdout")
    return json.loads(lines[-1])


def run_model(key: str, cfg: dict[str, Any], smiles: str, timeout: int) -> dict[str, Any]:
    command = [cfg["python"], abs_script_path(cfg["script"]), smiles, "both"]
    try:
        completed = subprocess.run(
            command,
            cwd=Path(cfg["cwd"]).expanduser(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"model": key, "error": f"timed out after {timeout} seconds"}
    except FileNotFoundError as exc:
        return {"model": key, "error": f"executable or cwd not found: {exc}"}
    except Exception as exc:
        return {"model": key, "error": str(exc)}

    try:
        payload = parse_runner_json(completed.stdout)
    except Exception as exc:
        stderr_tail = completed.stderr.strip().splitlines()[-1:] or []
        detail = f"could not parse runner JSON: {exc}"
        if stderr_tail:
            detail += f"; stderr: {stderr_tail[0]}"
        return {"model": key, "error": detail}

    if completed.returncode != 0 and "error" not in payload:
        stderr_tail = completed.stderr.strip().splitlines()[-1:] or []
        detail = stderr_tail[0] if stderr_tail else f"runner exited {completed.returncode}"
        return {"model": key, "error": detail}

    payload.setdefault("model", key)
    return payload


def group_peaks(shifts: list[dict[str, Any]], tolerance: float) -> list[dict[str, Any]]:
    values = sorted(
        float(row["shift_ppm"])
        for row in shifts
        if row.get("shift_ppm") is not None
    )
    if not values:
        return []

    groups: list[list[float]] = [[values[0]]]
    for value in values[1:]:
        center = sum(groups[-1]) / len(groups[-1])
        if abs(value - center) <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])

    peaks = [{"c": round(sum(group) / len(group), 4), "n": len(group)} for group in groups]
    return sorted(peaks, key=lambda peak: peak["c"], reverse=True)


def _real_couplings_source(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    fullsspruce = results.get("fullsspruce", {})
    if fullsspruce.get("error"):
        return []
    couplings = fullsspruce.get("couplings", [])
    return couplings if isinstance(couplings, list) else []


def merge_results(smiles: str, config: dict[str, Any], results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "smiles": smiles,
        "models": {},
        "spectra": {"C": {}, "H": {}},
    }
    real_couplings = _real_couplings_source(results)

    for key, result in results.items():
        cfg = config["models"][key]
        label = cfg.get("label", key)
        color = cfg.get("color", "#ffffff")
        error = result.get("error")
        couplings = result.get("couplings", [])
        if not isinstance(couplings, list):
            couplings = []
        merged["models"][key] = {
            "label": label,
            "color": color,
            "error": error,
            "couplings": bool(cfg.get("couplings", False)),
        }
        if error:
            continue

        shifts = result.get("shifts", [])
        if not isinstance(shifts, list):
            merged["models"][key]["error"] = "runner payload had non-list shifts"
            continue
        uses_real_couplings = key == "fullsspruce" and bool(couplings)
        annotation_couplings = couplings if uses_real_couplings else real_couplings
        annotated_shifts = annotate_shifts(
            smiles,
            shifts,
            annotation_couplings,
            uses_real_couplings=bool(annotation_couplings),
        )

        for nucleus in ("C", "H"):
            nucleus_shifts = [
                row
                for row in annotated_shifts
                if str(row.get("element", "")).upper() == nucleus
                and row.get("shift_ppm") is not None
            ]
            if not nucleus_shifts:
                continue
            spectrum_entry = {
                "label": label,
                "color": color,
                "peaks": group_peaks(nucleus_shifts, NUCLEUS_TOLERANCE[nucleus]),
                "shifts": nucleus_shifts,
            }
            if couplings:
                spectrum_entry["couplings"] = couplings
            merged["spectra"][nucleus][key] = spectrum_entry

    return merged


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(FRONTEND_PATH.read_text(encoding="utf-8"))


@app.get("/api/models")
def api_models() -> list[dict[str, Any]]:
    config = load_config()
    return [model_public_entry(key, cfg) for key, cfg in config.get("models", {}).items()]


@app.post("/api/structure")
def api_structure(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    kind = str(body.get("type", "")).strip().lower()
    value = str(body.get("value", ""))
    if kind not in {"smiles", "molblock", "name"}:
        raise HTTPException(status_code=400, detail="type must be one of smiles, molblock, name")
    result = canonicalize_structure(kind, value)
    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])
    return result


def selected_enabled_models(config: dict[str, Any], requested: Any) -> dict[str, dict[str, Any]]:
    models = config.get("models", {})
    if requested is None:
        keys = list(models.keys())
    elif isinstance(requested, list):
        keys = [str(key) for key in requested]
    else:
        raise HTTPException(status_code=400, detail="models must be a list of model keys")

    unknown = [key for key in keys if key not in models]
    if unknown:
        raise HTTPException(status_code=400, detail=f"unknown model key(s): {', '.join(unknown)}")

    return {
        key: models[key]
        for key in keys
        if bool(models[key].get("enabled", False))
    }


@app.post("/api/predict")
def api_predict(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    smiles = str(body.get("smiles", "")).strip()
    if not smiles:
        raise HTTPException(status_code=400, detail="smiles is required")

    config = load_config()
    timeout = int(config.get("timeout_seconds", 180))
    enabled = selected_enabled_models(config, body.get("models"))

    results: dict[str, dict[str, Any]] = {}
    if enabled:
        with ThreadPoolExecutor(max_workers=len(enabled)) as executor:
            futures = {
                executor.submit(run_model, key, cfg, smiles, timeout): key
                for key, cfg in enabled.items()
            }
            for future in as_completed(futures):
                key = futures[future]
                try:
                    results[key] = future.result()
                except Exception as exc:
                    results[key] = {"model": key, "error": str(exc)}

    merged = merge_results(smiles, config, results)
    merged["consensus"] = build_consensus(merged)
    merged["simulations"] = simulate_payload(merged)["spectra"]
    for nucleus in ("C", "H"):
        shifts = consensus_shifts(merged, merged["consensus"], nucleus)
        if shifts:
            merged["simulations"][nucleus]["consensus"] = {
                "label": "Consensus",
                "color": "#1A1A1A",
                "spectrum": simulate_spectrum(nucleus, shifts),
            }
    merged["hsqc"] = build_hsqc(merged)
    return merged


@app.post("/api/experimental")
def api_experimental(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    try:
        encoded = body.get("data_base64")
        data_bytes = None
        if encoded:
            try:
                data_bytes = base64.b64decode(str(encoded), validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("uploaded binary data is not valid base64") from exc
            if len(data_bytes) > 100 * 1024 * 1024:
                raise ValueError("experimental file exceeds the 100 MB limit")
        return parse_experimental(
            str(body.get("text", "")),
            str(body.get("nucleus", "")),
            str(body.get("filename", "")),
            data_bytes,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/simulate")
def api_simulate(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    if "spectra" not in body:
        raise HTTPException(status_code=400, detail="prediction payload with spectra is required")
    return simulate_payload(body)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
