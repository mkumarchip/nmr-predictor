import re
import io
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from consensus import build_consensus
from experimental import parse_experimental
from hsqc import build_hsqc


class ConsensusTests(unittest.TestCase):
    def test_alignment_weighting_and_disagreement(self):
        prediction = {
            "spectra": {
                "C": {
                    "first": {
                        "shifts": [
                            {"atom_index": 0, "shift_ppm": 18.0, "confidence": "High"}
                        ]
                    },
                    "second": {
                        "shifts": [
                            {"atom_index": 0, "shift_ppm": 22.0, "confidence": "Low"}
                        ]
                    },
                },
                "H": {},
            }
        }
        row = build_consensus(prediction)["C"][0]
        self.assertEqual(row["consensus_ppm"], 20.0)
        self.assertAlmostEqual(row["weighted_consensus_ppm"], 18.9231, places=4)
        self.assertEqual(row["spread_ppm"], 4.0)
        self.assertTrue(row["high_disagreement"])

    def test_fullsspruce_uses_inverse_standard_deviation_weight(self):
        prediction = {
            "spectra": {
                "H": {
                    "fullsspruce": {
                        "shifts": [{"atom_index": 3, "shift_ppm": 2.0, "std_ppm": 0.25}]
                    },
                    "csp5": {"shifts": [{"atom_index": 3, "shift_ppm": 1.0}]},
                },
                "C": {},
            }
        }
        row = build_consensus(prediction)["H"][0]
        self.assertEqual(row["consensus_ppm"], 1.5)
        self.assertEqual(row["weighted_consensus_ppm"], 1.8)


class ExperimentalTests(unittest.TestCase):
    def test_mnova_combined_report_with_assignments(self):
        text = (
            "1H NMR (400 MHz, CDCl3) δ 7.26 (d, J = 7.1 Hz, 2H), "
            "3.68 (q, J = 7.0 Hz, 2H), 1.20 (t, J = 7.1 Hz, 3H)\n"
            "13C NMR (101 MHz, CDCl3) δ 170.2, 128.4, 58.1, 18.3"
        )
        result = parse_experimental(text, "AUTO")
        self.assertEqual(result["format"], "mnova-report")
        self.assertEqual(result["frequency_mhz"], 400.0)
        self.assertEqual(result["solvent"], "CDCl3")
        self.assertEqual(result["spectra"]["H"]["peaks"][0]["multiplicity"], "d")
        self.assertEqual(result["spectra"]["H"]["peaks"][0]["J_hz"], [7.1])
        self.assertEqual(result["spectra"]["H"]["peaks"][0]["integration"], 2)
        self.assertEqual(
            [row["shift_ppm"] for row in result["spectra"]["C"]["peaks"]],
            [170.2, 128.4, 58.1, 18.3],
        )

    def test_mnova_range_and_multiple_j_values(self):
        result = parse_experimental(
            "1H NMR (500 MHz, DMSO-d6) δ 7.20–7.26 (dd, J = 7.1, 1.2 Hz, 1H)",
            "H",
        )
        peak = result["peaks"][0]
        self.assertEqual(peak["shift_ppm"], 7.23)
        self.assertEqual(peak["range_ppm"], [7.2, 7.26])
        self.assertEqual(peak["multiplicity"], "dd")
        self.assertEqual(peak["J_hz"], [7.1, 1.2])

    def test_plain_peak_list(self):
        result = parse_experimental("128.4\n77.0,0.4", "C")
        self.assertEqual([row["shift_ppm"] for row in result["peaks"]], [128.4, 77.0])
        self.assertIsNone(result["trace"])

    def test_jcamp_peak_table(self):
        text = """##TITLE=Test
##XFACTOR=1
##YFACTOR=1
##PEAK TABLE=(XY..XY)
7.26,1
1.20,0.5
##END=
"""
        result = parse_experimental(text, "H", "test.dx")
        self.assertEqual(len(result["peaks"]), 2)
        self.assertIsNone(result["trace"])
        self.assertEqual(result["parser"], "jcamp 1.3.2")

    def test_jcamp_full_trace_auto_nucleus_and_peak_pick(self):
        text = """##TITLE=Trace
##JCAMP-DX=5.00
##DATA TYPE=NMR SPECTRUM
##.OBSERVE NUCLEUS=^1H
##.OBSERVE FREQUENCY=400
##XUNITS=PPM
##YUNITS=ARBITRARY UNITS
##FIRSTX=3
##LASTX=1
##NPOINTS=5
##XFACTOR=1
##YFACTOR=1
##XYDATA=(X++(Y..Y))
3,0 0 1 0 0
##END=
"""
        result = parse_experimental(text, "AUTO", "trace.jdx")
        self.assertEqual(result["nucleus"], "H")
        self.assertEqual(result["peaks"], [{"shift_ppm": 2.0, "intensity": 1.0}])
        self.assertEqual(result["trace"]["x"], [3.0, 2.5, 2.0, 1.5, 1.0])

    def test_jcamp_hz_axis_converts_to_ppm(self):
        text = """##TITLE=Hz peaks
##JCAMP-DX=5.00
##.OBSERVE NUCLEUS=^1H
##.OBSERVE FREQUENCY=400
##XUNITS=HZ
##PEAK TABLE=(XY..XY)
2920,1
480,0.5
##END=
"""
        result = parse_experimental(text, "AUTO", "peaks.jdx")
        self.assertEqual([row["shift_ppm"] for row in result["peaks"]], [7.3, 1.2])

    def test_jeol_jdf_time_domain_pipeline(self):
        fid = np.exp(2j * np.pi * 0.1 * np.arange(64))
        dictionary = {
            "header": {
                "units": [(0, 1, "second")],
                "title": "Synthetic JEOL proton",
            }
        }
        universal = {
            "ndim": 1,
            0: {
                "size": 64,
                "sw": 4000.0,
                "obs": 400.0,
                "car": 1600.0,
                "label": "Proton",
            },
        }
        with (
            patch("nmrglue.jeol.read", return_value=(dictionary, fid)),
            patch("nmrglue.jeol.guess_udic", return_value=universal),
            patch("nmrglue.proc_autophase.autops", side_effect=lambda data, *args, **kwargs: data),
        ):
            result = parse_experimental("", "AUTO", "sample.jdf", b"JEOL-test")
        self.assertEqual(result["format"], "jeol-jdf")
        self.assertEqual(result["nucleus"], "H")
        self.assertTrue(result["auto_processed"])
        self.assertTrue(result["trace"]["x"])
        self.assertIn("auto-processed", result["processing_note"])

    def test_invalid_jeol_jdf_is_a_clear_parse_error(self):
        with self.assertRaisesRegex(ValueError, "could not read JEOL JDF file.*invalid or use an unsupported"):
            parse_experimental("", "H", "broken.jdf", b"not-a-jdf")

    def test_mnova_zip_extracts_embedded_jcamp(self):
        jcamp_text = """##TITLE=Embedded
##JCAMP-DX=5.00
##.OBSERVE NUCLEUS=^1H
##XUNITS=PPM
##PEAK TABLE=(XY..XY)
7.26,1
1.20,0.5
##END=
"""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("exports/spectrum.jdx", jcamp_text)
        result = parse_experimental("", "AUTO", "project.mnova", buffer.getvalue())
        self.assertEqual(result["format"], "mnova-container")
        self.assertEqual(result["embedded_file"], "exports/spectrum.jdx")
        self.assertEqual(len(result["peaks"]), 2)

    def test_native_mnova_returns_export_guidance(self):
        with self.assertRaisesRegex(ValueError, "Native \\.mnova files aren't supported"):
            parse_experimental("", "AUTO", "project.mnova", b"proprietary-data")


class HsqcTests(unittest.TestCase):
    def test_ethanol_equivalent_protons_group_to_two_cross_peaks(self):
        prediction = {
            "smiles": "CCO",
            "consensus": {
                "C": [
                    {"atom_index": 0, "weighted_consensus_ppm": 18.0},
                    {"atom_index": 1, "weighted_consensus_ppm": 58.0},
                ],
                "H": [
                    *[
                        {"atom_index": atom, "weighted_consensus_ppm": 1.2}
                        for atom in (3, 4, 5)
                    ],
                    *[
                        {"atom_index": atom, "weighted_consensus_ppm": 3.7}
                        for atom in (6, 7)
                    ],
                ],
            },
            "spectra": {"C": {}, "H": {}},
        }
        result = build_hsqc(prediction)
        self.assertEqual(result["source"], "consensus")
        self.assertEqual(len(result["peaks"]), 2)
        self.assertEqual(sorted(row["integration"] for row in result["peaks"]), [2, 3])


class OfflineFrontendTests(unittest.TestCase):
    def test_v3_controls_are_present_and_offline(self):
        html = (Path(__file__).parents[1] / "frontend" / "index.html").read_text()
        for marker in ("Consensus", "experimentalText", "assignmentTable", "2D HSQC"):
            self.assertIn(marker, html)
        for forbidden in ("localStorage", "sessionStorage"):
            self.assertNotIn(forbidden, html)
        self.assertIsNone(re.search(r"<(?:script|link)[^>]+(?:src|href)=[\"']https?://", html))


class StructureNumberingTests(unittest.TestCase):
    def test_drawing_uses_model_atom_order(self):
        from rdkit import Chem
        from structure import canonicalize_structure

        result = canonicalize_structure("smiles", "OCC")
        drawn = Chem.MolFromMolBlock(result["molblock"], removeHs=False)
        model = Chem.AddHs(Chem.MolFromSmiles(result["smiles"]))
        self.assertEqual(
            [atom.GetSymbol() for atom in drawn.GetAtoms()],
            [atom.GetSymbol() for atom in model.GetAtoms()],
        )


if __name__ == "__main__":
    unittest.main()
