#!/usr/bin/env python
import json
import os
import sys
import traceback


MODEL = "cascade"
PPM_SCALE = 50.484337
PPM_OFFSET = 99.798111


def atomic_number_tokenizer(atom):
    return atom.GetAtomicNum()


def emit(payload, exit_code=0):
    print(json.dumps(payload, separators=(",", ":")), flush=True)
    raise SystemExit(exit_code)


def confidence(delta_ppm):
    if delta_ppm >= 1.485774:
        return "Low"
    if delta_ppm >= 1.419831:
        return "Moderate"
    return "High"


def main():
    if len(sys.argv) < 2:
        emit({"model": MODEL, "error": "usage: run_cascade.py <SMILES> [both|13C]"}, 2)

    smiles = sys.argv[1]

    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        os.chdir("models/Predict_SMILES_FF_GPR")
        sys.path.insert(0, os.getcwd())
        sys.path.append("modules")

        import pickle
        import warnings

        import numpy as np
        import pandas as pd
        import tensorflow as tf
        from nfp.preprocessing import GraphSequence
        from rdkit import Chem
        from rdkit.Chem import AllChem

        from model import make_model

        warnings.filterwarnings("ignore")
        tf.get_logger().setLevel("ERROR")
        tf.keras.backend.set_floatx("float64")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"invalid SMILES: {smiles}")
        mol = Chem.AddHs(mol)
        embed_code = AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
        if embed_code != 0:
            raise RuntimeError(f"RDKit EmbedMolecule failed with code {embed_code}")
        opt_code = AllChem.MMFFOptimizeMolecule(mol)
        if opt_code < 0:
            raise RuntimeError(f"RDKit MMFFOptimizeMolecule failed with code {opt_code}")

        carbon_indices = np.array(
            [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6]
        ).astype(int)
        if len(carbon_indices) == 0:
            emit({"model": MODEL, "shifts": []})

        inp_df = pd.DataFrame({"Mol": [mol], "atom_index": [carbon_indices]})

        def _compute_stacked_offsets(sizes, repeats):
            return np.repeat(np.cumsum(np.hstack([0, sizes[:-1]])), repeats)

        def ragged_const(inp_arr):
            return tf.ragged.constant(np.expand_dims(inp_arr, axis=0), ragged_rank=1)

        def mol_iter(df):
            for _, row in df.iterrows():
                yield row["Mol"], row["atom_index"]

        class RBFSequence(GraphSequence):
            def process_data(self, batch_data):
                offset = _compute_stacked_offsets(batch_data["n_pro"], batch_data["n_atom"])
                offset = np.where(batch_data["atom_index"] >= 0, offset, 0)
                batch_data["atom_index"] += offset

                features = [
                    "node_attributes",
                    "node_coordinates",
                    "edge_indices",
                    "atom_index",
                    "n_pro",
                ]
                for feature in features:
                    batch_data[feature] = ragged_const(batch_data[feature])

                del batch_data["n_atom"]
                del batch_data["n_bond"]
                del batch_data["distance"]
                del batch_data["bond"]
                del batch_data["node_graph_indices"]

                return batch_data

        with open("preprocessor_orig.p", "rb") as fh:
            input_data = pickle.load(fh)
        preprocessor = input_data["preprocessor"]
        inputs_test = preprocessor.predict(mol_iter(inp_df))
        test_sequence = RBFSequence(inputs_test, batch_size=32)

        model = make_model()
        model.load_weights("best_model_val_mae.h5")

        predictions = []
        uncertainty = []
        for x in test_sequence:
            distribution = model(x)
            predictions.extend(distribution.mean().numpy().flatten())
            uncertainty.extend(distribution.stddev().numpy().flatten())

        shifts = []
        for atom_index, pred, stddev in zip(carbon_indices, predictions, uncertainty):
            shift_ppm = float(pred) * PPM_SCALE + PPM_OFFSET
            pred_ub = (float(pred) + 1.96 * float(stddev)) * PPM_SCALE + PPM_OFFSET
            shifts.append(
                {
                    "element": "C",
                    "atom_index": int(atom_index),
                    "shift_ppm": round(shift_ppm, 2),
                    "confidence": confidence(pred_ub - shift_ppm),
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
