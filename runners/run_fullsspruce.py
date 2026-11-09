#!/usr/bin/env python
"""FullSSPrUCe runner. Runs INSIDE the `fullsspruce` conda env, cwd
~/fullsspruce-public-master. Predicts 1H shifts and scalar couplings.

Prints JSON:
  {"model":"fullsspruce",
   "shifts":[{"element":"H","atom_index":3,"shift_ppm":3.35,"confidence":null}, ...],
   "couplings":[{"atom_i":6,"atom_j":7,"J_hz":6.99}, ...]}   # H-H only

Usage: python run_fullsspruce.py "SMILES" [both]
"""
import json
import os
import sys

# FullSSPrUCe checkpoints predate PyTorch 2.6's weights_only=True default.
# This env var forces the old behaviour so torch.load can unpickle the model.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

# Only H-H couplings matter for 1H multiplet drawing. Drop one-bond C-H /
# C-C couplings (large) and negligible long-range ones (near zero).
MIN_ABS_J_HZ = 0.5
MAX_ABS_J_HZ = 20.0


def main():
    smiles = sys.argv[1]

    from rdkit import Chem
    from rdkit.Chem import AllChem
    from fullsspruce import predictor

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Bad SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=0xF00D) != 0:
        AllChem.EmbedMolecule(mol, useRandomCoords=True)
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        AllChem.UFFOptimizeMolecule(mol)

    is_h = {atom.GetIdx(): (atom.GetSymbol() == "H") for atom in mol.GetAtoms()}

    p = predictor.Predictor()
    out = p.predict([mol], properties=["1H", "coupling"])
    records = out[0] if isinstance(out, tuple) else out
    rec = records[0]

    shifts = []
    for row in rec.get("1H", []):
        shifts.append(
            {
                "element": "H",
                "atom_index": int(row["atom_idx"]),
                "shift_ppm": round(float(row["pred_mu"]), 3),
                "std_ppm": round(float(row["pred_std"]), 4),
                "confidence": None,
            }
        )
    couplings = []
    for row in rec.get("coupling", []):
        i = int(row["atom_idx1"])
        j = int(row["atom_idx2"])
        j_hz = float(row["coup_mu"])
        if not (is_h.get(i) and is_h.get(j)):
            continue
        if not (MIN_ABS_J_HZ <= abs(j_hz) <= MAX_ABS_J_HZ):
            continue
        couplings.append({"atom_i": i, "atom_j": j, "J_hz": round(j_hz, 2)})

    print(json.dumps({"model": "fullsspruce", "shifts": shifts, "couplings": couplings}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"model": "fullsspruce", "error": str(exc)}))
        sys.exit(1)
