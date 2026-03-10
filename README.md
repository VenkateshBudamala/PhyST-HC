# 🌊 PhyST-HC Framework

![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red.svg)
![Graph Learning](https://img.shields.io/badge/Graph-GNN-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

# 🚀 Overview

**PhyST-HC (Physics-guided Hybrid Spatial–Temporal Framework)** is a hybrid deep learning framework developed for **event-scale prediction of streamflow and riverine carbon fluxes**.

The framework integrates:

- 🌍 **Process-based hydrological simulations (SWAT)**
- 🧠 **Spatial Graph Neural Networks (GCN)**
- ⏳ **Temporal learning for dynamic hydrological processes**

to improve prediction of:

- **Streamflow (Q)**
- **Dissolved Organic Carbon (DOC)**
- **Particulate Organic Carbon (POC)**

This hybrid framework combines **physical process understanding and data-driven learning** to improve prediction accuracy across hydrological events.

---

# 🧩 Repository Structure

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

# 📂 Scripts Description

| File | Description |
|-----|-------------|
| `Inputs.py` | Model configuration, experiment parameters, and data paths |
| `Main_Module.py` | Core PhyST-HC model architecture |
| `Run_Simulation.py` | Main training and testing pipeline |
| `Functions_Q.py` | Streamflow prediction functions |
| `Functions_DOC.py` | Dissolved Organic Carbon prediction functions |
| `Functions_POC.py` | Particulate Organic Carbon prediction functions |

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/PhyST-HC.git
cd PhyST-HC
```

Install required Python packages:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Framework

1. Open the configuration file:

```
scripts/Inputs.py
```

Update all required inputs including:

- Data paths  
- Model hyperparameters  
- Training settings  
- Output directories  

2. After updating the inputs, run the main simulation script:

```bash
python scripts/Run_Simulation.py
```

The framework will automatically:

- Load the configured inputs
- Train and test the models
- Run simulations for all variables (**Q, DOC, and POC**)
- Save predicted outputs to the specified output directory

---

# 📊 Observed Data Format

The observed data file specified in **`excel_path`** inside `Inputs.py` must follow the format described below.

---

# Excel File Structure

The observed Excel file must contain **three separate sheets**:

| Sheet Name | Variable |
|-------------|----------|
| `Q` | Streamflow observations |
| `POC` | Particulate Organic Carbon observations |
| `DOC` | Dissolved Organic Carbon observations |

Each variable must be stored in **its corresponding sheet**.

---

# Sheet Format

Each sheet must follow the structure below:

| Date | 1 | 2 | 3 | 4 | 5 |
|-----|---|---|---|---|---|
| 2001-01-01 | value | value | value | value | value |
| 2001-01-02 | value | value | value | value | value |

Where:

- **First column must be `Date`**
- Remaining columns represent **observed station numbers**

---

# 🔗 Observed–Reach Mapping

In `Inputs.py`, the mapping between **observed station columns and SWAT reach numbers** should be defined as:

```python
obs_to_reach_mapping = {
    1:1, 2:2, 3:3, 4:4, 5:5,
    6:6, 7:7, 8:8, 9:9, 10:10,
    11:11, 12:12, 13:13, 14:14, 15:15,
    16:16, 17:17, 18:18, 19:19
}
```

Where:

- **Key → Observed column number in the Excel sheet**
- **Value → Corresponding SWAT reach number**

Example:

```
Observed column 1 → SWAT Reach 1
Observed column 2 → SWAT Reach 2
Observed column 3 → SWAT Reach 3
```

---

# Important Notes

- The **Date column must be the first column** in each sheet.
- Sheet names must be exactly:

```
Q
POC
DOC
```

- Observed station columns must match the numbers used in `obs_to_reach_mapping`.
- Missing values should be represented as **NaN or empty cells**.

---

# 📦 Requirements

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
- openpyxl  

Install them using:

```bash
pip install -r requirements.txt
```

---

# 🔬 Framework Concept

PhyST-HC integrates **process-based hydrological modeling with deep learning architectures** to capture both:

- Spatial dependencies across river networks
- Temporal dynamics of hydrological and biogeochemical processes

This enables improved prediction of **streamflow and carbon flux transport** in river systems.

---

# 📊 Applications

The framework can be applied to:

- Event-scale streamflow prediction  
- Riverine carbon transport modeling  
- Physics-guided machine learning for hydrology  
- Hydrological forecasting systems  

---

# 👨‍🔬 Author

**Venkatesh Budamala**  
Gwangju Institute of Science and Technology (GIST)

Hydrology • AI for Earth Systems • Hybrid Modeling
