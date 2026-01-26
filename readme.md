# 🧬 CADD Toolkit for CB2 Agonists

This repository provides a collection of tools for **Computer-Aided Drug Design (CADD)** focused on **CB2 receptor agonists**. The project integrates **large language models (LLMs)** for molecule generation with a **GNN-based QSAR model** for affinity prediction, along with auxiliary utilities for chemical structure processing.

---

## 📦 Repository Contents

### 🔹 `ChemBERT_module2.py`
Module implementing the **first LLM**, based on the ChemBERTa architecture, used for working with SMILES representations.

**Source model:** ChemBERTaLM  
https://huggingface.co/gokceuludogan/ChemBERTaLM

---


### 🔹 `affinity_predictor.py`
A **CLI script** for predicting binding affinity (`pKi`) toward CB2 using a trained QSAR model.

Features:
- prediction for **single SMILES strings**
- batch prediction for **`.csv` files** (provided as a string path argument)

The script relies on the included PyTorch model.

---

### 🔹 `attentivefp_model_final_full_data.pth`
A trained **QSAR model** in PyTorch:
- **GNN-based architecture** (Graph Attention / AttentiveFP)
- trained on the full dataset
- used to predict **pKi** values for CB2 agonists

---

### 🔹 `gen-gnnVS.ipynb`
A research notebook integrating the full CADD workflow:
- molecule generation using **LLMs**
- `pKi` prediction using the GNN model
- calculation of molecular descriptors
- selection of candidates for downstream analysis (virtual screening)

The notebook serves as a **generative + predictive CADD pipeline**.

---

### 🔹 `openbabel_converter.py`
A **CLI utility** based on OpenBabel:
- converts **SMILES → `.pdb` files**
- input: a `.csv` file containing SMILES (passed as a path argument)

Generated `.pdb` files can be directly used in **PyMOL**, **AutoDock**, **Vina**, or other docking software.

---

### 🔹 `drugGen_generator.py`
Script implementing the **second LLM**, which generates molecules based on **biological (protein) sequences**.

**Source model:** DrugGen  
https://huggingface.co/alimotahharynia/DrugGen

---


## 🧠 Summary
The project combines:
- **LLM-based molecular generation** (ChemBERTaLM, DrugGen)
- **GNN-based QSAR modeling** for affinity (`pKi`) prediction
- **CLI tools** for prediction and structure conversion

Altogether, it forms a complete **generative + predictive CADD pipeline** for **CB2 agonists**.

---

## 📚 References

If you use this repository or its components, please cite the original works associated with the underlying models:

Sheikholeslami, M., Mazrouei, N., Gheisari, Y., Fasihi, A., Irajpour, M., & Motahharynia, A.* (2025).  
DrugGen enhances drug discovery with large language models and reinforcement learning.  
*Scientific Reports*, 15, 13445. https://doi.org/10.1038/s41598-025-98629-1

```bibtex
@article{Sheikholeslami2025DrugGen,
  title   = {DrugGen enhances drug discovery with large language models and reinforcement learning},
  author  = {Sheikholeslami, M. and Mazrouei, N. and Gheisari, Y. and Fasihi, A. and Irajpour, M. and Motahharynia, A.},
  journal = {Scientific Reports},
  volume  = {15},
  pages   = {13445},
  year    = {2025},
  doi     = {10.1038/s41598-025-98629-1}
}
```

Uludoğan, G., Ozkirimli, E., Ulgen, K. O., Karalı, N. L., & Özgür, A. (2022).  
Exploiting Pretrained Biochemical Language Models for Targeted Drug Design.  
*Bioinformatics*. https://doi.org/10.1093/bioinformatics/btac482

```bibtex
@article{Uludogan2022ChemBERTa,
  title   = {Exploiting Pretrained Biochemical Language Models for Targeted Drug Design},
  author  = {Uludoğan, Gökçe and Ozkirimli, Elif and Ulgen, Kutlu O. and Karalı, Nilgün Lütfiye and Özgür, Arzucan},
  journal = {Bioinformatics},
  year    = {2022},
  doi     = {10.1093/bioinformatics/btac482}
}
```

---

## 🧾 Citation of this repository

If you use **tools, pipelines, or models provided in this repository** (e.g. for CB2 agonist discovery or related molecular design tasks), please cite the repository author as:

> **Adam Mazur** — *CADD tools for CB2 agonist discovery* (GitHub repository)

This is an **informal citation** intended for acknowledgements, software citations, or methods sections when the repository is used as a research tool.

---




## ⚠️ Disclaimer
- The repository integrates pretrained models from the literature; proper citation of the original works is required when used in academic contexts.
- The code is intended for **research and proof-of-concept** purposes.

---

