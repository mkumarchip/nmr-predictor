#!/usr/bin/env python
import json
import sys
import traceback


MODEL = "csp5"


def emit(payload, exit_code=0):
    print(json.dumps(payload, separators=(",", ":")), flush=True)
    raise SystemExit(exit_code)


def wanted_nuclei(arg):
    value = (arg or "both").lower()
    if value in {"both", "all"}:
        return [("13C", "C"), ("1H", "H")]
    if value in {"13c", "c", "carbon"}:
        return [("13C", "C")]
    if value in {"1h", "h", "proton"}:
        return [("1H", "H")]
    return [("13C", "C"), ("1H", "H")]


def main():
    if len(sys.argv) < 2:
        emit({"model": MODEL, "error": "usage: run_csp5.py <SMILES> [both|13C|1H]"}, 2)

    smiles = sys.argv[1]
    nuclei = wanted_nuclei(sys.argv[2] if len(sys.argv) > 2 else "both")

    try:
        import csp5

        shifts = []
        for nucleus, expected_element in nuclei:
            result = csp5.predict_smiles(
                smiles,
                nucleus=nucleus,
                device="auto",
                num_conformers=1,
            )
            payload = result.to_compact_mapped_payload()
            for molecule in payload.get("molecules", []):
                for row in molecule.get("predictions", []):
                    element = str(row.get("element") or expected_element).upper()
                    if element not in {"C", "H"}:
                        continue
                    atom_map = row.get("atom_map")
                    if atom_map is None:
                        raise ValueError(f"CSP5 row missing atom_map: {row}")
                    shifts.append(
                        {
                            "element": element,
                            "atom_index": int(atom_map) - 1,
                            "shift_ppm": float(row["shift_ppm"]),
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
