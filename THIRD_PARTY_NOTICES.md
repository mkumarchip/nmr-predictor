# Third-Party Models and Libraries

This repository does **not** include any prediction model code, trained weights, or
datasets. Each model is installed separately by the user into its own environment and
is called through the small runner scripts in `runners/`. Point `config.json` at those
installs (see `config.example.json`).

Please follow each project's license and cite the original papers when you use their
predictions.

## Prediction models

| Model | Authors | Code license | How to get it |
|-------|---------|--------------|---------------|
| CSP5 | Benji Rowlands | MIT | `pip install csp5` (https://pypi.org/project/csp5/) |
| CASCADE-2.0 | Abhijeet Bhadauria & Robert Paton / Paton Lab | MIT | CASCADE-2.0 repository from the Paton Lab, including its pretrained models |
| NMRNet | AI4EC (Xu et al.) | MIT | https://github.com/Colin-Jay/NMRNet, weights from Zenodo (below) |
| FullSSPrUCe | The Jonas Lab, University of Chicago | MIT (declared in `setup.py`) | https://github.com/thejonaslab/fullsspruce-public |

Notes:

- **NMRNet weights and data** (https://zenodo.org/records/19142375,
  DOI 10.5281/zenodo.19142375) are licensed **CC BY 4.0**, which requires credit to the
  creators. Cite: Xu et al., *Nature Computational Science* 5, 292–300 (2025),
  DOI 10.1038/s43588-025-00783-z.
- **FullSSPrUCe** declares MIT in its package metadata but its repository has no
  LICENSE file. Cite: *Rapid Prediction of Full Spin Systems using Uncertainty-Aware
  Machine Learning*, DOI 10.1039/D3SC01930F.
- **CASCADE-2.0**: `runners/run_cascade.py` reuses the output scaling constants and
  confidence thresholds from CASCADE-2.0's prediction code.
  Copyright (c) 2025 Abhijeet Bhadauria & Robert Paton / Paton Lab, MIT License.

## Libraries

| Library | License | Used for |
|---------|---------|----------|
| RDKit | BSD-3-Clause | structure handling, 2D depiction |
| FastAPI | MIT | web server |
| Uvicorn | BSD-3-Clause | web server |
| nmrglue | BSD | JEOL JDF import |
| jcamp | MIT | JCAMP-DX import |
| OPSIN / py2opsin | MIT | optional offline name-to-structure |

OPSIN asks users to cite: Lowe et al., *J. Chem. Inf. Model.* 2011, 51(3), 739–753,
DOI 10.1021/ci100384d.

The MIT License requires its copyright and permission notice to be kept with copies of
the licensed software. Those notices ship with each package when you install it.
