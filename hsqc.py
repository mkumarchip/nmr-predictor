from typing import Any

from rdkit import Chem


def _shift_map(rows: list[dict[str, Any]], field: str) -> dict[int, float]:
    return {int(row["atom_index"]): float(row[field]) for row in rows if row.get(field) is not None}


def build_hsqc(prediction: dict[str, Any]) -> dict[str, Any]:
    smiles = prediction.get("smiles", "")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"source": None, "peaks": []}
    mol = Chem.AddHs(mol)

    consensus = prediction.get("consensus", {})
    h_map = _shift_map(consensus.get("H", []), "weighted_consensus_ppm")
    c_map = _shift_map(consensus.get("C", []), "weighted_consensus_ppm")
    source = "consensus"

    if not h_map or not c_map:
        source = None
        for key in ("fullsspruce", "csp5", "nmrnet"):
            h_rows = prediction.get("spectra", {}).get("H", {}).get(key, {}).get("shifts", [])
            c_rows = prediction.get("spectra", {}).get("C", {}).get(key, {}).get("shifts", [])
            if h_rows and c_rows:
                h_map = _shift_map(h_rows, "shift_ppm")
                c_map = _shift_map(c_rows, "shift_ppm")
                source = key
                break

    raw = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1 or atom.GetIdx() not in h_map:
            continue
        carbon = next((neighbor for neighbor in atom.GetNeighbors() if neighbor.GetAtomicNum() == 6), None)
        if carbon is None or carbon.GetIdx() not in c_map:
            continue
        raw.append(
            {
                "h_ppm": h_map[atom.GetIdx()],
                "c_ppm": c_map[carbon.GetIdx()],
                "h_atoms": [atom.GetIdx()],
                "c_atom": carbon.GetIdx(),
            }
        )

    groups: list[dict[str, Any]] = []
    for peak in raw:
        existing = next(
            (
                item
                for item in groups
                if item["c_atom"] == peak["c_atom"]
                and abs(item["h_ppm"] - peak["h_ppm"]) <= 0.03
                and abs(item["c_ppm"] - peak["c_ppm"]) <= 0.5
            ),
            None,
        )
        if existing:
            existing["h_atoms"].extend(peak["h_atoms"])
            count = len(existing["h_atoms"])
            existing["h_ppm"] = round(
                (existing["h_ppm"] * (count - 1) + peak["h_ppm"]) / count, 4
            )
        else:
            groups.append(dict(peak))

    for peak in groups:
        peak["h_atoms"] = sorted(peak["h_atoms"])
        peak["h_ppm"] = round(peak["h_ppm"], 4)
        peak["c_ppm"] = round(peak["c_ppm"], 4)
        peak["integration"] = len(peak["h_atoms"])
    return {"source": source, "peaks": groups}
