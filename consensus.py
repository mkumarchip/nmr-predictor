import math
from collections import defaultdict
from typing import Any


CONFIDENCE_WEIGHTS = {"High": 1.0, "Moderate": 0.6, "Low": 0.3}
DISAGREEMENT_THRESHOLDS = {"C": 3.0, "H": 0.3}


def _weight(model_key: str, row: dict[str, Any]) -> float:
    if model_key == "fullsspruce":
        std_ppm = row.get("std_ppm")
        if std_ppm is not None and float(std_ppm) > 0:
            return 1.0 / float(std_ppm)
    return CONFIDENCE_WEIGHTS.get(str(row.get("confidence")), 1.0)


def build_consensus(prediction: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Align successful model shifts by RDKit atom index and summarize them."""
    output: dict[str, list[dict[str, Any]]] = {"C": [], "H": []}
    for nucleus in ("C", "H"):
        by_atom: dict[int, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
        for model_key, model_data in prediction.get("spectra", {}).get(nucleus, {}).items():
            for row in model_data.get("shifts", []):
                if row.get("atom_index") is None or row.get("shift_ppm") is None:
                    continue
                by_atom[int(row["atom_index"])].append((model_key, row))

        for atom_index, values in sorted(by_atom.items()):
            # A consensus with one contributor would conceal rather than measure uncertainty.
            if len(values) < 2:
                continue
            shifts = [float(row["shift_ppm"]) for _, row in values]
            mean = sum(shifts) / len(shifts)
            weights = [_weight(model_key, row) for model_key, row in values]
            weighted_mean = sum(value * weight for value, weight in zip(shifts, weights)) / sum(weights)
            variance = sum((value - mean) ** 2 for value in shifts) / len(shifts)
            spread = max(shifts) - min(shifts)
            output[nucleus].append(
                {
                    "atom_index": atom_index,
                    "element": nucleus,
                    "consensus_ppm": round(mean, 4),
                    "weighted_consensus_ppm": round(weighted_mean, 4),
                    "spread_ppm": round(spread, 4),
                    "stdev_ppm": round(math.sqrt(variance), 4),
                    "n_models": len(values),
                    "per_model": {key: round(float(row["shift_ppm"]), 4) for key, row in values},
                    "high_disagreement": spread > DISAGREEMENT_THRESHOLDS[nucleus],
                }
            )
    return output


def consensus_shifts(
    prediction: dict[str, Any], consensus: dict[str, list[dict[str, Any]]], nucleus: str
) -> list[dict[str, Any]]:
    """Create simulation-ready shifts while preserving real-J annotations when available."""
    spectra = prediction.get("spectra", {}).get(nucleus, {})
    preferred_keys = ["fullsspruce", "csp5", "nmrnet", "cascade"]
    templates: dict[int, dict[str, Any]] = {}
    for key in preferred_keys + [key for key in spectra if key not in preferred_keys]:
        for row in spectra.get(key, {}).get("shifts", []):
            atom_index = int(row.get("atom_index", -1))
            if atom_index >= 0 and atom_index not in templates:
                templates[atom_index] = row

    shifts = []
    for row in consensus.get(nucleus, []):
        atom_index = int(row["atom_index"])
        item = dict(templates.get(atom_index, {}))
        item.update(
            {
                "element": nucleus,
                "atom_index": atom_index,
                "shift_ppm": float(row["weighted_consensus_ppm"]),
                "confidence": None,
                "consensus": True,
            }
        )
        shifts.append(item)
    return shifts
