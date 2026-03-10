# 🌊 PhyST-HC Framework

![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)
![Graph Learning](https://img.shields.io/badge/Graph-GNN-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🚀 Overview

**PhyST-HC** (Physics-guided Hybrid Spatial–Temporal Framework) is a deep learning framework designed for **event-scale prediction of streamflow and riverine carbon fluxes**.

The framework integrates:

- 🌍 **Process-based hydrological simulations (SWAT)**
- 🧠 **Spatial Graph Neural Networks (GCN)**
- ⏳ **Temporal learning using attention mechanisms**

to improve prediction of:

- **Streamflow (Q)**
- **Dissolved Organic Carbon (DOC)**
- **Particulate Organic Carbon (POC)**

This hybrid approach combines **physics-based modeling and AI** to improve predictive skill in hydrological systems.

---

## 🧩 Repository Structure

```
PhyST-HC
│
├── README.md
├── requirements.txt
├── LICENSE
│
└── scripts
    ├── Inputs.py
    ├── Main_Module.py
    ├── Run_Simulation.py
    ├── Functions_Q.py
    ├── Functions_DOC.py
    └── Functions_POC.py
```

---

## 📂 scripts

| File | Description |
|-----|-------------|
| `Inputs.py` | Model configuration and experiment parameters |
| `Main_Module.py` | Core PhyST-HC architecture |
| `Run_Simulation.py` | Main training and testing pipeline |
| `Functions_Q.py` | Streamflow prediction module |
| `Functions_DOC.py` | Dissolved Organic Carbon prediction module |
| `Functions_POC.py` | Particulate Organic Carbon prediction module |

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/PhyST-HC.git
cd PhyST-HC
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Model

Run the main simulation script:

```bash
python scripts/Run_Simulation.py
```

---

## 📦 Requirements

The framework uses the following Python libraries:

- torch  
- torch-geometric  
- numpy  
- pandas  
- scikit-learn  
- hydroeval  
- geopandas  
- matplotlib  
- tqdm  

---

## 🔬 Framework Concept

PhyST-HC integrates **process-based hydrological modeling with deep learning architectures** to capture both:

- Spatial dependencies across river networks
- Temporal dynamics of hydrological events

This enables improved prediction of **hydrological and biogeochemical fluxes** in river systems.

---

## 📊 Applications

- Event-scale streamflow prediction  
- Riverine carbon transport modeling  
- Physics-guided machine learning for hydrology  
- Hydrological forecasting systems  

---

## 👨‍🔬 Author

**Venkatesh Budamala**  
Gwangju Institute of Science and Technology (GIST)

Hydrology • AI for Earth Systems • Hybrid Modeling
