#!/usr/bin/env python
import json
import os
import sys
import traceback


MODEL = "nmrnet"


def emit(payload, exit_code=0):
    print(json.dumps(payload, separators=(",", ":")), flush=True)
    raise SystemExit(exit_code)


def element_from_nucleus(nucleus):
    text = str(nucleus).lower()
    if "h" in text:
        return "H"
    if "c" in text:
        return "C"
    return None


def get_calibration(calib, nucleus):
    aliases = {
        "13C": ["13C", "C", "carbon", "Carbon"],
        "1H": ["1H", "H", "proton", "Proton"],
    }.get(nucleus, [nucleus])
    for key in aliases:
        if key in calib:
            return key, calib[key]
    return None, None


def normalize_rows(result, element):
    if isinstance(result, dict) and isinstance(result.get("shifts"), list):
        rows = result["shifts"]
    elif isinstance(result, tuple) and len(result) == 2:
        indices, raws = result
        rows = [
            {"atom_index": atom_index, "raw": raw}
            for atom_index, raw in zip(indices, raws)
        ]
    elif isinstance(result, list):
        rows = result
    elif isinstance(result, dict):
        rows = []
        for key, value in result.items():
            try:
                atom_index = int(key)
                raw = float(value)
            except (TypeError, ValueError):
                continue
            rows.append({"atom_index": atom_index, "raw": raw})
    else:
        rows = []

    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        atom_index = row.get("atom_index", row.get("atom_idx", row.get("idx")))
        raw = row.get("raw", row.get("prediction", row.get("pred", row.get("shift", row.get("shift_ppm")))))
        if atom_index is None or raw is None:
            continue
        normalized.append(
            {
                "element": row.get("element", element),
                "atom_index": int(atom_index),
                "raw": float(raw),
            }
        )
    return normalized


def main():
    if len(sys.argv) < 2:
        emit({"model": MODEL, "error": "usage: run_nmrnet.py <SMILES> [both|13C|1H]"}, 2)

    smiles = sys.argv[1]
    requested = (sys.argv[2] if len(sys.argv) > 2 else "both").lower()
    nucleus_order = []
    if requested in {"both", "all", "13c", "c", "carbon"}:
        nucleus_order.append("C")
    if requested in {"both", "all", "1h", "h", "proton"}:
        nucleus_order.append("H")

    try:
        sys.path.insert(0, os.getcwd())
        import nmrnet_predict

        calib = nmrnet_predict._load_calib()
        shifts = []
        for nucleus in nucleus_order:
            calib_key, params = get_calibration(calib, nucleus)
            if not params:
                continue
            element = element_from_nucleus(calib_key or nucleus)
            if element is None:
                continue
            args = nmrnet_predict.build_args(nucleus)
            from unicore.data import Dictionary

            dictionary = Dictionary.load(args.dict_path)
            dictionary.add_symbol("[MASK]", is_special=True)
            model = nmrnet_predict.load_model(args, dictionary)
            result = nmrnet_predict.run_model_on_smiles(
                smiles,
                nucleus,
                args,
                dictionary,
                model,
            )
            a = float(params["a"])
            b = float(params["b"])
            for row in normalize_rows(result, element):
                shifts.append(
                    {
                        "element": element,
                        "atom_index": row["atom_index"],
                        "shift_ppm": a + b * row["raw"],
                        "confidence": None,
                    }
                )

        emit({"model": MODEL, "shifts": shifts})
    except Exception as exc:
        message = f"{exc.__class__.__name__}: {exc}"
        if not str(exc):
            message = traceback.format_exc(limit=1).strip()
        emit({"model": MODEL, "error": message}, 1)


if __name__ == "__main__":
    main()
