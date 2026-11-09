from typing import Any


WINDOWS = {
    "H": {"min": -1.0, "max": 13.0, "step": 0.002, "frequency_mhz": 400.0, "linewidth_hz": 1.0},
    "C": {"min": -10.0, "max": 220.0, "step": 0.04, "frequency_mhz": 100.0, "linewidth_hz": 3.0},
}


def _axis(nucleus: str) -> list[float]:
    win = WINDOWS[nucleus]
    n = int(round((win["max"] - win["min"]) / win["step"])) + 1
    return [round(win["max"] - i * win["step"], 6) for i in range(n)]


def _split_lines(center_ppm: float, groups: list[dict[str, Any]], frequency_mhz: float) -> list[tuple[float, float]]:
    lines = [(float(center_ppm), 1.0)]
    for group in groups:
        j_hz = abs(float(group.get("J_hz", 0.0)))
        count = max(1, int(group.get("n", 1)))
        if j_hz <= 0:
            continue
        for _ in range(count):
            delta_ppm = j_hz / frequency_mhz / 2.0
            next_lines = []
            for ppm, intensity in lines:
                next_lines.append((ppm - delta_ppm, intensity / 2.0))
                next_lines.append((ppm + delta_ppm, intensity / 2.0))
            lines = next_lines
    return lines


def simulate_spectrum(nucleus: str, shifts: list[dict[str, Any]]) -> dict[str, Any]:
    if nucleus not in WINDOWS:
        raise ValueError(f"unsupported nucleus: {nucleus}")

    win = WINDOWS[nucleus]
    x = _axis(nucleus)
    gamma_ppm = (win["linewidth_hz"] / win["frequency_mhz"]) / 2.0
    y = [0.0 for _ in x]
    annotations = []

    for shift in shifts:
        center = float(shift["shift_ppm"])
        integration = float(shift.get("integration", 1) or 1)
        groups = shift.get("coupling_groups") or [
            {"J_hz": j, "n": 1} for j in shift.get("J_hz", [])
        ]
        if nucleus == "C":
            groups = []
        lines = _split_lines(center, groups, float(win["frequency_mhz"]))
        for line_ppm, weight in lines:
            for i, ppm in enumerate(x):
                dx = ppm - line_ppm
                y[i] += integration * weight * gamma_ppm * gamma_ppm / (dx * dx + gamma_ppm * gamma_ppm)
        annotations.append(
            {
                "atom_index": shift.get("atom_index"),
                "ppm": center,
                "multiplicity": shift.get("multiplicity", "s"),
                "J_hz": shift.get("J_hz", []),
                "integration": integration,
                "estimated": bool(shift.get("estimated", False)),
                "lines": [{"ppm": round(ppm, 6), "intensity": round(weight, 6)} for ppm, weight in lines],
            }
        )

    max_y = max(y) if y else 0.0
    if max_y > 0:
        y = [round(value / max_y, 6) for value in y]

    return {
        "nucleus": nucleus,
        "x": x,
        "y": y,
        "window": win,
        "annotations": annotations,
    }


def simulate_payload(prediction: dict[str, Any]) -> dict[str, Any]:
    spectra: dict[str, Any] = {"H": {}, "C": {}}
    for nucleus, models in prediction.get("spectra", {}).items():
        if nucleus not in spectra:
            continue
        for key, model_data in models.items():
            spectra[nucleus][key] = {
                "label": model_data.get("label", key),
                "color": model_data.get("color", "#ffffff"),
                "spectrum": simulate_spectrum(nucleus, model_data.get("shifts", [])),
            }
    return {"smiles": prediction.get("smiles"), "spectra": spectra}
