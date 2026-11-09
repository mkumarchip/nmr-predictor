from collections import defaultdict
from typing import Any

from rdkit import Chem


DEFAULT_J_HZ = 7.0
PROTON_TOLERANCE = 0.03
CARBON_TOLERANCE = 0.5

MULTIPLICITY_BY_COUNT = {
    0: "s",
    1: "d",
    2: "t",
    3: "q",
    4: "quint",
}


def multiplicity_from_count(n: int) -> str:
    return MULTIPLICITY_BY_COUNT.get(n, "m")


def _attached_carbon(mol: Chem.Mol, atom_idx: int) -> int | None:
    atom = mol.GetAtomWithIdx(atom_idx)
    if atom.GetAtomicNum() != 1:
        return None
    for neighbor in atom.GetNeighbors():
        if neighbor.GetAtomicNum() == 6:
            return neighbor.GetIdx()
    return None


def n_plus_one_for_proton(mol: Chem.Mol, atom_idx: int) -> dict[str, Any]:
    carbon_idx = _attached_carbon(mol, atom_idx)
    if carbon_idx is None:
        return {"multiplicity": "s", "J_hz": [], "estimated": True}

    carbon = mol.GetAtomWithIdx(carbon_idx)
    n = 0
    for neighbor in carbon.GetNeighbors():
        if neighbor.GetAtomicNum() != 6:
            continue
        for candidate in neighbor.GetNeighbors():
            if candidate.GetAtomicNum() == 1:
                n += 1

    return {
        "multiplicity": multiplicity_from_count(n),
        "J_hz": [DEFAULT_J_HZ] if n > 0 else [],
        "coupling_groups": [{"J_hz": DEFAULT_J_HZ, "n": n}] if n > 0 else [],
        "estimated": True,
    }


def _group_close(values: list[float], tolerance: float = 0.6) -> list[dict[str, Any]]:
    groups: list[list[float]] = []
    for value in sorted(abs(v) for v in values if abs(v) >= 0.5):
        if not groups or abs(value - (sum(groups[-1]) / len(groups[-1]))) > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [{"J_hz": round(sum(group) / len(group), 3), "n": len(group)} for group in groups]


def _name_from_coupling_groups(groups: list[dict[str, Any]]) -> str:
    if not groups:
        return "s"
    labels = [multiplicity_from_count(int(group["n"])) for group in groups]
    if len(labels) == 1:
        return labels[0]
    return "".join(label[0] if label != "quint" else "quint" for label in labels)


def _proton_coupling_groups(
    mol: Chem.Mol,
    atom_idx: int,
    couplings: list[dict[str, Any]],
    shift_by_atom: dict[int, float],
) -> list[dict[str, Any]]:
    values = []
    if _attached_carbon(mol, atom_idx) is None:
        return []
    for coupling in couplings:
        atom_i = int(coupling.get("atom_i", -1))
        atom_j = int(coupling.get("atom_j", -1))
        if atom_idx not in {atom_i, atom_j}:
            continue
        partner = atom_j if atom_i == atom_idx else atom_i
        if partner < 0 or partner >= mol.GetNumAtoms():
            continue
        if mol.GetAtomWithIdx(partner).GetAtomicNum() != 1:
            continue
        if _attached_carbon(mol, partner) is None:
            continue
        if abs(shift_by_atom.get(atom_idx, 1e9) - shift_by_atom.get(partner, -1e9)) <= PROTON_TOLERANCE:
            continue
        j_hz = float(coupling.get("J_hz", 0.0))
        if 0.5 <= abs(j_hz) <= 20.0:
            values.append(j_hz)
    return _group_close(values)


def _integration_map(shifts: list[dict[str, Any]], element: str) -> dict[int, int]:
    tolerance = PROTON_TOLERANCE if element == "H" else CARBON_TOLERANCE
    indexed = sorted(
        [
            (int(row["atom_index"]), float(row["shift_ppm"]))
            for row in shifts
            if row.get("element") == element and row.get("shift_ppm") is not None
        ],
        key=lambda item: item[1],
    )
    integration: dict[int, int] = {}
    groups: list[list[tuple[int, float]]] = []
    for item in indexed:
        if not groups:
            groups.append([item])
            continue
        center = sum(v for _, v in groups[-1]) / len(groups[-1])
        if abs(item[1] - center) <= tolerance:
            groups[-1].append(item)
        else:
            groups.append([item])
    for group in groups:
        for atom_idx, _ in group:
            integration[atom_idx] = len(group)
    return integration


def annotate_shifts(
    smiles: str,
    shifts: list[dict[str, Any]],
    couplings: list[dict[str, Any]] | None = None,
    uses_real_couplings: bool = False,
) -> list[dict[str, Any]]:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if mol is None:
        return shifts

    couplings = couplings or []
    h_integration = _integration_map(shifts, "H")
    c_integration = _integration_map(shifts, "C")
    h_shift_by_atom = {
        int(row["atom_index"]): float(row["shift_ppm"])
        for row in shifts
        if row.get("element") == "H" and row.get("shift_ppm") is not None
    }
    annotated = []
    for row in shifts:
        item = dict(row)
        element = str(item.get("element", "")).upper()
        atom_idx = int(item.get("atom_index", -1))
        if element == "H" and 0 <= atom_idx < mol.GetNumAtoms():
            if uses_real_couplings:
                groups = _proton_coupling_groups(mol, atom_idx, couplings, h_shift_by_atom)
                item["coupling_groups"] = groups
                item["J_hz"] = [group["J_hz"] for group in groups]
                item["multiplicity"] = _name_from_coupling_groups(groups)
                item["estimated"] = False
            else:
                item.update(n_plus_one_for_proton(mol, atom_idx))
            item["integration"] = h_integration.get(atom_idx, 1)
        elif element == "C":
            item["multiplicity"] = "s"
            item["J_hz"] = []
            item["coupling_groups"] = []
            item["integration"] = c_integration.get(atom_idx, 1)
            item["estimated"] = not uses_real_couplings
        annotated.append(item)
    return annotated


def couplings_by_atom(couplings: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    by_atom: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for coupling in couplings:
        by_atom[int(coupling["atom_i"])].append(coupling)
        by_atom[int(coupling["atom_j"])].append(coupling)
    return by_atom
