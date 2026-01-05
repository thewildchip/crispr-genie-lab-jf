# crispr-genie-lab
CRISPR gRNA Efficiency & Off-Target Prediction

This repository contains our **Jugend forscht 2025/26** project. We develop machine-learning models to

* **predict gRNA on-target efficiency** and
* **estimate off-target risk**

using genomic, epigenetic, and sequence-based features.

---

## Project Goals

* Build **reproducible ML pipelines** (data → features → models → evaluation)
* Compare **classical ML** and **neural network** approaches
* Understand **which biological features matter most** for CRISPR performance
* Keep the code **clean, modular** and get the **best possible outcome** as a two-man team

---

## Repository Structure

(not implemented yet)

---

## Workflow

1. **Raw data** stays unchanged in `data/raw/`
2. Cleaning & feature engineering → `data/processed/`
3. Experiments happen in `notebooks/`
4. All reusable logic lives in `src/mypackage`

---

## Our latest models

* **RandomForrestRegressions and HistGradientBoostingRegressor** for gRNA efficiency

## Our future models

* **Sequence-based + epigenetic features**
* Baselines (linear models, tree-based models)
* Neural networks (when justified)

---

## Tech Stack

* Python (NumPy, pandas, scikit-learn)
* PyTorch (coming in future)
* Jupyter Notebook
* Git & GitHub

---

## Notes

* Final results and conclusions will be documented for **Jugend forscht submission**

---

## Authors

Jugend forscht team (Carl-Orff-Gymnasium Bayern, 2025/26)
Tobias Weichelt & Daniel Panoor
