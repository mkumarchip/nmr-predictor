import os
import html
import subprocess
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem, Draw


COMMON_NAME_SMILES = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "acetylsalicylic acid": "CC(=O)Oc1ccccc1C(=O)O",
    "ethanol": "CCO",
    "acetic acid": "CC(=O)O",
    "benzene": "c1ccccc1",
    "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
}


def _mol_from_name(name: str) -> Chem.Mol | None:
    text = name.strip()
    if not text:
        return None
    fallback = COMMON_NAME_SMILES.get(text.lower())
    if fallback:
        return Chem.MolFromSmiles(fallback)

    try:
        import py2opsin

        smiles = py2opsin.py2opsin(text)
        if smiles:
            return Chem.MolFromSmiles(smiles)
    except Exception:
        pass

    jar = _find_opsin_jar()
    if jar:
        try:
            completed = subprocess.run(
                ["java", "-jar", str(jar), "-osmi", text],
                capture_output=True,
                text=True,
                timeout=15,
            )
            smiles = completed.stdout.strip().splitlines()[-1].strip()
            if completed.returncode == 0 and smiles:
                return Chem.MolFromSmiles(smiles)
        except Exception:
            pass

    return None


def _find_opsin_jar() -> Path | None:
    candidates = []
    if os.environ.get("OPSIN_JAR"):
        candidates.append(Path(os.environ["OPSIN_JAR"]).expanduser())
    candidates += [
        Path(__file__).resolve().parent / "opsin.jar",
        Path.home() / "opsin.jar",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _mol_from_input(kind: str, value: str) -> Chem.Mol | None:
    kind = kind.lower().strip()
    if kind == "smiles":
        return Chem.MolFromSmiles(value)
    if kind == "name":
        return _mol_from_name(value)
    if kind == "molblock":
        first_record = value.split("$$$$", 1)[0].strip()
        mol = Chem.MolFromMolBlock(first_record, sanitize=True, removeHs=False)
        if mol is not None:
            return mol
        return Chem.MolFromMol2Block(first_record, sanitize=True, removeHs=False)
    raise ValueError(f"unsupported structure type: {kind}")


def _prepare_2d_mol(mol: Chem.Mol) -> Chem.Mol:
    prepared = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(prepared)
    except Exception:
        pass
    prepared = Chem.AddHs(prepared)
    AllChem.Compute2DCoords(prepared)
    return prepared


def canonicalize_structure(kind: str, value: str) -> dict[str, Any]:
    if not value or not str(value).strip():
        return {"error": "empty structure input"}

    mol = _mol_from_input(kind, value)
    if mol is None:
        return {"error": f"could not parse {kind}: {html.escape(str(value)[:80])}"}

    heavy = Chem.RemoveHs(Chem.Mol(mol))
    canonical_smiles = Chem.MolToSmiles(heavy, canonical=True, isomericSmiles=True)
    # Models index atoms in canonical-SMILES order, so the drawing must too.
    display_mol = _prepare_2d_mol(Chem.MolFromSmiles(canonical_smiles))
    for atom in display_mol.GetAtoms():
        atom.SetProp("atomNote", str(atom.GetIdx()))
    molblock = Chem.MolToMolBlock(display_mol)
    svg = Draw.MolsToGridImage(
        [display_mol],
        molsPerRow=1,
        subImgSize=(420, 300),
        useSVG=True,
    )
    if isinstance(svg, bytes):
        svg = svg.decode("utf-8")

    return {
        "smiles": canonical_smiles,
        "molblock": molblock,
        "svg": svg,
        "num_atoms": int(display_mol.GetNumAtoms()),
        "error": None,
    }

