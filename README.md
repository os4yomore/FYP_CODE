# FYP_CODE
Repository for My Final Year Project

# Beyond Standardised Protocols

**A Safety-Constrained Reinforcement Learning Framework for Managing Adaptive Response to Pain in Sickle Cell Disease**

Final-year undergraduate research project — BSc Computer Science, Pan-Atlantic University (2026).

---

## Overview

This project implements an offline reinforcement learning system for opioid dosing decisions during vaso-occlusive crises (VOC) in sickle cell disease (SCD) patients. The core contribution is a two-layer safety architecture that combines Conservative Q-Learning (CQL) with hard clinical constraint filtering, demonstrating that safety-constrained learning can match the performance of tolerance-aware oracle baselines while operating only on observable clinical state.

The system is evaluated entirely in-silico, consistent with established practice in healthcare reinforcement learning (Gottesman et al., 2019).

---

## What's In This Repository

| File / Folder | Description |
|---|---|
| `app.py` | Streamlit clinician-facing decision support interface |
| `scd_rl_project.ipynb` | Main Jupyter notebook — simulator, dataset generation, baselines, CQL training, evaluation |
| `cql_agent.pth` | Trained CQL agent weights (PyTorch state dict) |
| `scd_offline_dataset.csv` | Offline dataset of 2,000 patient trajectories (~59,000 transitions) |
| `requirements.txt` | Python dependencies |
| `figures/` | Validation and results plots |

---

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/scd-pain-management-rl.git
cd scd-pain-management-rl
pip install -r requirements.txt
```

### Run the Streamlit Interface

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

### Reproduce the Results

Open the Jupyter notebook and run all cells in order:

```bash
jupyter notebook scd_rl_project.ipynb
```

The notebook generates the simulator, the offline dataset, trains the CQL agent, and evaluates against three baselines on the held-out test set.

---

## System Architecture

The system consists of four major components:

1. **Patient Simulator** — phenomenological simulator producing synthetic VOC trajectories across three patient phenotypes (mild, moderate, severe)
2. **Offline Dataset** — 2,000 patient hospitalisations generated using a mixed behaviour policy
3. **Baseline Policies** — random, standard clinical protocol, and a tolerance-aware oracle heuristic
4. **CQL Agent** — Conservative Q-Learning agent with structural safety constraint filtering at inference time

---

## Headline Results

Evaluated on a held-out test set of 300 patients across 5 training seeds:

| Policy | Mean Pain (NRS) | % at Target | Safety Violations |
|---|---|---|---|
| Random | 3.99 | 35.4% | 0 |
| Standard Protocol | 4.29 | 17.7% | 0 |
| Reactive Heuristic (oracle) | 3.92 | 30.5% | 0 |
| **CQL Agent (5-seed mean)** | **3.95 ± 0.15** | **28.9% ± 4.2%** | **0 / 1500** |

The CQL agent significantly outperforms current clinical practice (the standard protocol) and matches the oracle reactive heuristic while operating only on observable clinical inputs, with zero safety violations across 1,500 test episodes.

---

## Key References

- Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020). Conservative Q-learning for offline reinforcement learning. *NeurIPS, 33*, 1179–1191.
- Levine, S., Kumar, A., Tucker, G., & Fu, J. (2020). Offline reinforcement learning: Tutorial, review, and perspectives on open problems. *arXiv:2005.01643*.
- Gottesman, O., et al. (2019). Guidelines for reinforcement learning in healthcare. *Nature Medicine, 25*(1), 16–18.
- Sagi, V., Mittal, A., Tran, H., & Gupta, K. (2021). Pain in sickle cell disease: Current and potential translational therapies. *Pain Research and Management, 2021*, 5569829.

---

## Important Note

This is a **proof-of-concept research project**, not a clinically validated system. The simulator is phenomenological rather than mechanistic, and no patient data was used in training. The work is intended to demonstrate the technical feasibility of safety-constrained offline RL for healthcare decision support, not to be deployed in any clinical setting.

---

## Author

**Ruth Osayomore Olotu** — BSc Computer Science, Pan-Atlantic University, 2026.
