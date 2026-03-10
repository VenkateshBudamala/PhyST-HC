# ==============================================================================
## 1. Required Packages
# ==============================================================================


import os
import random
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from hydroeval import nse, kge, pbias

warnings.filterwarnings("ignore")

# --- Hardware Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("Using device:", device)


# ================= Functions =================
import os
import shutil
from pathlib import Path

def create_save_dir_and_save_scripts(
    project_root, Loss, seq_len_list, hidden_trans, hidden_gcn, learning_rate, LR_method, transformations
):
    # ── concise folder name ──
    save_dir = os.path.join(
        project_root,
        f"{transformations}_{LR_method}_{Loss}_s{seq_len_list[0]}_T{hidden_trans}_G{hidden_gcn}_LR{learning_rate}"
    )
    os.makedirs(save_dir, exist_ok=True)

    # ── save scripts ──
    scripts_dir = Path(save_dir) / "scripts"
    scripts_dir.mkdir(exist_ok=True)

    code_dir = Path(os.getcwd())  # Code folder

    for f in ["Run_Simulation.py", "Inputs.py", "Functions_DOC.py", "Main_Module.py"]:
        src = code_dir / f
        if src.exists():
            shutil.copy(src, scripts_dir / f)

    return save_dir


import sys
from pathlib import Path

class ConsoleLogger:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def start_console_logging(save_dir):
    log_file = Path(save_dir) / "DOC_console.log"
    logger = ConsoleLogger(log_file)
    sys.stdout = logger
    sys.stderr = logger
    return logger


def stop_console_logging(logger):
    logger.close()
    sys.stdout = logger.terminal
    sys.stderr = logger.terminal


# ==============================================================================
## 2. Model & Loss Definitions
# ==============================================================================

class PositionalEncoding(nn.Module):
    """Adds fixed sinusoidal positional encoding to an input tensor."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term[:len(pe[:, 0::2][0])])
        pe[:, 1::2] = torch.cos(position * div_term[:len(pe[:, 1::2][0])])

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, n_features).
        Returns:
            Tensor: Input tensor with positional encoding added.
        """
        return x + self.pe[:, :x.size(1), :]


class ImprovedTransformerGCN(nn.Module):
    """
    Hybrid model combining a Transformer Encoder for temporal feature extraction
    and a GCN for spatial feature propagation.
    """
    def __init__(self, n_features, hidden_trans=128, hidden_gcn=256, seq_len=14,
                 dropout=0.2, num_heads=2, trans_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.pos_encoder = PositionalEncoding(n_features, max_len=seq_len)

        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=n_features, nhead=num_heads,
            dim_feedforward=hidden_trans, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=trans_layers)

        # GCN Layers
        self.gcn1 = GCNConv(n_features, hidden_gcn)
        self.gcn2 = GCNConv(hidden_gcn, hidden_gcn)
        # Residual connection if feature dimensions differ
        self.residual = nn.Linear(n_features, hidden_gcn) if n_features != hidden_gcn else nn.Identity()

        # Output layers
        self.out_conv = GCNConv(hidden_gcn, 1)
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_gcn)
        self.bn2 = nn.BatchNorm1d(hidden_gcn)

    def forward(self, data):
        """
        Args:
            data (torch_geometric.data.Data): Graph data object.
        Returns:
            Tensor: Predicted streamflow (untransformed).
        """
        x, edge_index = data.x, data.edge_index
        # x shape: [n_nodes * n_samples, seq_len * n_features_per_time]
        n_nodes_samples = x.shape[0]
        n_features_per_time = x.shape[1] // self.seq_len
        
        # Reshape to sequence format: [n_nodes*n_samples, seq_len, n_features_per_time]
        x_seq = x.view(n_nodes_samples, self.seq_len, n_features_per_time)

        # Pos encoding + Transformer
        x_seq = self.pos_encoder(x_seq)
        x_trans = self.transformer_encoder(x_seq)
        # Take last time-step representation for GCN: [n_nodes*n_samples, n_features_per_time]
        x_trans = x_trans[:, -1, :]

        # GCN block with residual connection
        residual = self.residual(x_trans)
        x = F.relu(self.bn1(self.gcn1(x_trans, edge_index)))
        x = self.dropout(x)
        # GCNConv output is added to residual *before* final activation/dropout, 
        # but here the BN is applied to GCN1 output. Following original code logic:
        x = F.relu(self.bn2(self.gcn2(x, edge_index)) + residual)
        x = self.dropout(x)
        x = self.out_conv(x, edge_index)
        return x.squeeze()


class HydroGraphLoss(nn.Module):
    def __init__(self, alpha=0.6, beta=0.25, gamma=0.15, eps=1e-8):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.eps = alpha, beta, gamma, eps

    def forward(self, pred, obs):
        pred, obs = pred.view(-1), obs.view(-1)

        q90 = torch.quantile(obs, 0.90)
        weights = torch.where(obs >= q90, 5, 1.0)

        rmse = torch.sqrt(torch.mean(weights * (pred - obs)**2) + self.eps)

        peak_mask = obs >= q90
        if peak_mask.sum() > 5:
            corr = torch.corrcoef(torch.stack([pred[peak_mask], obs[peak_mask]]))[0, 1]
            trend_penalty = 1 - corr
        else:
            trend_penalty = 0.0

        bias_penalty = torch.abs(torch.mean(pred) - torch.mean(obs)) / (torch.std(obs) + self.eps)

        return self.alpha * rmse + self.beta * trend_penalty + self.gamma * bias_penalty


class DifferentialHydroLoss(nn.Module):
    """
    Hydrology-aware differential loss:
    - Value loss (weighted RMSE)
    - Differential loss (dQ/dt)
    - Peak emphasis
    """
    def __init__(self, alpha=0.5, beta=0.35, gamma=0.15, eps=1e-6):
        super().__init__()
        self.alpha = alpha  # magnitude loss
        self.beta = beta    # differential loss
        self.gamma = gamma  # bias
        self.eps = eps

    def forward(self, pred, obs):
        pred = pred.view(-1)
        obs = obs.view(-1)

        # -------------------------------
        # 1. Peak-weighted RMSE
        # -------------------------------
        q90 = torch.quantile(obs, 0.9)
        weights = torch.where(obs >= q90, 5.0, 1.0)

        rmse = torch.sqrt(
            torch.mean(weights * (pred - obs) ** 2) + self.eps
        )

        # -------------------------------
        # 2. Differential loss (dQ/dt)
        # -------------------------------
        d_pred = pred[1:] - pred[:-1]
        d_obs  = obs[1:]  - obs[:-1]

        # emphasize rising limbs only
        rising_mask = d_obs > 0

        if rising_mask.sum() > 5:
            diff_loss = torch.mean(
                (d_pred[rising_mask] - d_obs[rising_mask]) ** 2
            )
        else:
            diff_loss = torch.mean((d_pred - d_obs) ** 2)

        # -------------------------------
        # 3. Bias (water balance)
        # -------------------------------
        bias_loss = torch.abs(pred.mean() - obs.mean()) / (obs.std() + self.eps)

        # -------------------------------
        # Final loss
        # -------------------------------
        return (
            self.alpha * rmse +
            self.beta  * diff_loss +
            self.gamma * bias_loss
        )


class KGELoss(nn.Module):
    """
    KGE-based loss function for hydrological model optimization.
    Minimizes 1 - KGE (Kling–Gupta Efficiency), so higher KGE => lower loss.
    """

    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, eps=1e-8, device="cpu"):
        """
        Parameters:
            alpha: Weight for correlation component (r)
            beta:  Weight for variability ratio component (σ_pred/σ_obs)
            gamma: Weight for bias ratio component (μ_pred/μ_obs)
            eps:   Numerical stability constant
            device: Device to run computations on
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        self.device = device

    def forward(self, pred: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1).to(self.device)
        obs = obs.view(-1).to(self.device)

        # --- Means and standard deviations ---
        mu_pred = torch.mean(pred)
        mu_obs = torch.mean(obs)
        std_pred = torch.std(pred)
        std_obs = torch.std(obs)

        # --- Correlation (r) ---
        cov = torch.mean((pred - mu_pred) * (obs - mu_obs))
        r = cov / (std_pred * std_obs + self.eps)

        # --- Variability ratio (α = std_pred / std_obs) ---
        alpha_ratio = std_pred / (std_obs + self.eps)

        # --- Bias ratio (β = mu_pred / mu_obs) ---
        beta_ratio = mu_pred / (mu_obs + self.eps)

        # --- Kling–Gupta Efficiency ---
        kge = 1 - torch.sqrt(
            (self.alpha * (r - 1)) ** 2 +
            (self.beta * (alpha_ratio - 1)) ** 2 +
            (self.gamma * (beta_ratio - 1)) ** 2
        )

        # --- Loss (1 - KGE): higher KGE = lower loss ---
        loss = 1 - kge
        return loss




# ==============================================================================
## 3. Utility & Feature Engineering Functions
# ==============================================================================
def inverse_log_transform(data: np.ndarray, offset: float = 1e-8) -> np.ndarray:
    """Convert back from log space to original scale."""
    return np.exp(data) - offset


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day_of_year, sin/cos seasonality, month, year."""
    df = df.copy()
    if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['day_of_year'] = df['Date'].dt.dayofyear
        df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        df['month'] = df['Date'].dt.month
        df['year'] = df['Date'].dt.year
    return df


def select_relevant_features(df_list: list[pd.DataFrame], target_col: str = 'Obs_DOC',
                             importance_threshold: float = 0.05, 
                             drop_candidates: list[str] = None, features_to_add: list[str] = None) -> tuple[list[str], pd.DataFrame]:
    """
    Select features with RandomForest importance > threshold
    """
    combined = pd.concat(df_list, ignore_index=True)
    combined = combined.dropna(subset=[target_col])
    
    if drop_candidates is None:
        drop_candidates = ['Date', 'Obs_DOC', 'year', 'day_of_year','FLOW_OUTcms','GCN_DOC']

    feature_cols = [c for c in combined.columns if c not in drop_candidates]
    if len(feature_cols) == 0:
        raise ValueError("No candidate features available for selection.")

    X = combined[feature_cols].values
    y = combined[target_col].values

    rf = RandomForestRegressor(n_estimators=1000, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importances = rf.feature_importances_
    
    importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
    importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)

    importance_df.to_excel(os.path.join(save_dir, "DOC_Feature_Importance_RF.xlsx"), index=False)
    
    selected_features = importance_df[importance_df['Importance'] > importance_threshold]['Feature'].tolist()



    # List only the features you want to check in this style
    features_to_add = features_to_add
    
    for feat in features_to_add:
        if feat in combined.columns and feat not in selected_features:
            selected_features.append(feat)


    print(f"Selected features > {importance_threshold}: {len(selected_features)} features")
    # print(selected_features)

    return selected_features, importance_df



def create_lag_features(
    df_list: list[pd.DataFrame],
    seq_len: int,
    selected_features: list[str] = None
):
    """
    Create lagged sequences for each reach for BOTH:
        - Original values
        - Log-transformed values (log1p)

    Returns:
        {
            "original": (features, targets, masks),
            "log":      (features_log, targets_log, masks_log)
        }
    """

    def make_lags(df_list, seq_len, selected_features, log_flag):

        feats, targs, masks = [], [], []

        for df in df_list:
            df = df.copy().reset_index(drop=True)

            # Select features
            if selected_features is not None:
                feature_cols = [c for c in selected_features if c in df.columns]
            else:
                feature_cols = [
                    c for c in df.columns
                    if c not in ['Date', 'Obs_DOC', 'day_of_year', 'month', 'year', 'GCN_DOC']
                ]

            if not feature_cols:
                raise ValueError("No valid selected features found!")

            X = df[feature_cols].values.astype(float)
            y = df["Obs_DOC"].values.astype(float)
            mask = ~np.isnan(y)

            if log_flag:
                X = np.log1p(X)
                y = np.log1p(y)

            # Build lagged sequences
            X_seq = []
            for i in range(seq_len, len(df)):
                X_seq.append(X[i-seq_len:i, :])

            if len(X_seq) == 0:
                feats.append(np.zeros((0, seq_len, len(feature_cols))))
                targs.append(np.zeros((0,)))
                masks.append(np.zeros((0,), dtype=bool))
            else:
                feats.append(np.stack(X_seq))
                targs.append(y[seq_len:])
                masks.append(mask[seq_len:])

        return feats, targs, masks

    # Return both
    return {
        "original": make_lags(df_list, seq_len, selected_features, log_flag=False),
        "log":      make_lags(df_list, seq_len, selected_features, log_flag=True)
    }



def plot_hydrograph(results_df: pd.DataFrame, save_dir: str, station_id: str = None):
    """
    Plot hydrograph with:
    - Observed Q as scatter
    - Predicted Q as line
    - Inverted precipitation
    - Date-sorted (CRITICAL)
    """

    # ─── 1. SORT BY DATE (VERY IMPORTANT) ───
    results_df = results_df.sort_values("Date").reset_index(drop=True)

    plt.figure(figsize=(13, 6))
    ax1 = plt.gca()

    # ─── 2. OBSERVED: SCATTER (no line) ───
    ax1.scatter(
        results_df["Date"],
        results_df["Obs(kg/day)"],
        color="black",
        s=10,
        alpha=0.5,
        label="Observed Flow",
        zorder=3
    )

    # ─── 3. PREDICTED: LINE ───
    ax1.plot(
        results_df["Date"],
        results_df["Pred(kg/day)"],
        color="blue",
        linestyle="-",
        linewidth=1,
        label="Predicted Flow",
        zorder=2
    )

    ax1.set_ylabel("Log-DOC (kg/day)")
    ax1.set_xlabel("Date")
    ax1.legend(loc="upper left")

    # ─── 4. PRECIPITATION (INVERTED) ───
    ax2 = ax1.twinx()
    ax2.bar(
        results_df["Date"],
        results_df["PRECIPmm"],
        width=1.0,
        color="skyblue",
        alpha=0.5,
        label="Precipitation",
        zorder=1
    )
    ax2.set_ylabel("Precipitation (mm)")
    ax2.invert_yaxis()
    ax2.legend(loc="upper right")

    # ─── 5. TITLE & SAVE ───
    title = "Hydrograph with Precipitation"
    if station_id:
        title += f" – {station_id}"
    plt.title(title)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"DOC_hydrograph_{station_id}.png"),
        dpi=300
    )
    plt.close()

def plot_scatter(obs: np.ndarray, pred: np.ndarray, title: str, save_path: str):
    """Plot Observed vs. Predicted scatter plot."""
    plt.figure(figsize=(8, 8))
    plt.scatter(obs, pred, alpha=0.6)
    if len(obs) > 0 and len(pred) > 0:
        max_val = max(np.max(obs), np.max(pred))
        plt.plot([0, max_val], [0, max_val], 'r--')
    plt.xlabel('Observed')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ==============================================================================
## 4. Data Loading & Graph Construction Functions
# ==============================================================================


def load_Parameter_Set(swat_path: str, excel_path: str, obs_to_reach_mapping: dict,
                       shapefile_path: str = None, SWAT_dates:str = list, 
                       dates: pd.DatetimeIndex = None,
                       warmup_years: int = 2, seq_len: int = 5,
                       save_dir = str) -> list[pd.DataFrame]:
    
    """Loads and preprocesses SWAT output, shapefile attributes, and observed data."""
    print("Loading SWAT data and attributes...")
    # output_sub = np.genfromtxt(os.path.join(swat_path, 'output.sub'), skip_header=9)
    output_rch = np.genfromtxt(os.path.join(swat_path, 'output.rch'), skip_header=9)
   
    output_sub = np.genfromtxt(os.path.join(swat_path, 'output.sub'), skip_header=9, usecols=range(1, 70))

    
    # Find and read shapefiles
    def find_shp(folder, prefix):
        files = [f for f in os.listdir(folder) if f.lower().startswith(prefix) and f.endswith(".shp")]
        if not files:
            raise FileNotFoundError(f"No shapefile starting with '{prefix}' found in {folder}")
        return gpd.read_file(os.path.join(folder, files[0]))

    df_outlet = find_shp(shapefile_path, "monitor")
    df_watershed = find_shp(shapefile_path, "subs")

    # Attribute Preprocessing (merging outlet/watershed attributes)
    subset_df_outlet = df_outlet[["Type", "Subbasin"]].copy()
    subset_df_outlet["Type_rank"] = subset_df_outlet["Type"].apply(lambda x: 0 if x == "R" else 1)
    subset_df_outlet = subset_df_outlet.sort_values("Type_rank").drop_duplicates("Subbasin", keep="first")
    subset_df_outlet = subset_df_outlet.drop(columns="Type_rank")
    subset_df_outlet["Subbasin"] = subset_df_outlet["Subbasin"].astype(int)

    subset_df_watershed = pd.DataFrame(df_watershed).iloc[:, 2:15].copy()
    
    if 'Long_' in subset_df_watershed.columns:
        subset_df_watershed['Long_'] = subset_df_watershed['Long_'].abs()
        
    subset_df_watershed["Subbasin"] = subset_df_watershed["Subbasin"].astype(int)
    merged_attr_df = pd.merge(subset_df_outlet, subset_df_watershed, on="Subbasin", how="left").sort_values("Subbasin").reset_index(drop=True)

    # SWAT output processing

    # Select Subbasin + required SWAT variables (by column positions)
    cols = [0] + list(range(3, 3+45))   # adjust 45 if your file has fewer/more columns
    output_sub_df = pd.DataFrame(output_sub).iloc[:, cols]
    
    output_sub_df.columns = [
        "Subbasin", "AREAkm2", "PRECIPmm", "SNOMELTmm", "PETmm", "ETmm", "SWmm", "PERCmm", "SURQmm", "GW_Qmm",
        "WYLDmm", "SYLDt/ha", "ORGNkg/ha", "ORGPkg/ha", "SEDPkg/ha", "LATQmm", "GWNO3kg/ha", "TNO3kg/ha",
        "QTILEmm", "FRNO3kg/ha", "FRNH4kg/ha", "S_NO3kg/ha", "S_NH3kg/ha",
        "SedCkg/ha", "SurfCkg/ha", "LatCkg/ha", "PercCkg/ha",
        "NPPCkg/ha", "RspCkg/ha",
        "SnoFallmm", "SnoDepthmm", "SnoWatermm",
        "AirTempC",
        "SolT50cmC", "SolT100cmC", "SolT150cmC", "SolT200cmC", "SolT300cmC", "SolT350cmC", "SoT1000cmC",
        "FrozeDay",
        "SWCSol_TotalC", "Sol_TotaS",
        "wtmp_SurQ", "wtmp_LatQ", "wtmp_GwQ"
    ]
    
    selected_cols = [
        "Subbasin",
        "AREAkm2",
    
        # Hydrologic forcing
        "PRECIPmm",
        "SNOMELTmm",
    
        # Subsurface transport (dominant for DOC)
        "GW_Qmm",
        "PERCmm",
        "LATQmm",
        "QTILEmm",
        'SURQmm',
        'WYLDmm',
    
        # Storage & concentration control
        "SWmm",
        "ETmm",
    
        # Soil carbon sources
        "SWCSol_TotalC",
        "Sol_TotaS",
        "NPPCkg/ha",
        "RspCkg/ha",
    
        # Temperature control
        "AirTempC"
    ]      ###### for doc


    output_sub_df = output_sub_df[selected_cols].copy()

    
    # Type fix
    output_sub_df["Subbasin"] = output_sub_df["Subbasin"].astype(int)
    
    # Merge attributes
    output_sub_with_attr = output_sub_df.merge(merged_attr_df, on="Subbasin", how="left")
    
 
    # Combine RCH flow with SUB attributes
    output_rch_df = pd.DataFrame(output_rch).iloc[:, [5,6,10,13,15,27,24]]
    output_rch_df.columns = ["FLOW_INcms","FLOW_OUTcms",
                              "SED_OUTtons", "ORGN_OUTkg","ORGP_OUTkg",
                              "CBOD_OUTkg","CHLA_OUTkg" ]
    
    # output_rch_df = pd.DataFrame(output_rch).iloc[:, [5,6]]
    # output_rch_df.columns = ["FLOW_INcms","FLOW_OUTcms"]

    output_final = output_sub_with_attr.copy()
    output_final = pd.concat([output_final, output_rch_df],axis=1)


    n_subbasins = len(output_sub_df["Subbasin"].unique())
    n_days = len(output_sub_df) // n_subbasins
    if dates is None:
        dates = pd.date_range(start=SWAT_dates[0], periods=n_days, freq="D")
    date_column = np.repeat(dates, n_subbasins)
    output_final.insert(0, "Date", date_column)
    
  
    output_final["Month"] = output_final["Date"].dt.month
    if 'Type' in output_final.columns:
        output_final["Type"] = output_final["Type"].map({"T": 0, "R": 1})
    else:
        output_final["Type"] = 0
        
    output_final["Long_"] = output_final["Long_"].abs()
    output_final["Lat"] = output_final["Lat"].abs()

    
    print("Loading observed data...")
    df_obs = pd.read_excel(excel_path, sheet_name='DOC')
    df_obs['Date'] = pd.to_datetime(df_obs.iloc[:, 0])
    df_obs = df_obs.set_index('Date')
    df_obs.replace(-9999, np.nan, inplace=True)
    # ---- LOG TRANSFORM OBSERVED VALUES (SAFE) ----
    df_obs = np.log1p(df_obs.clip(lower=0))


    # Map observation IDs to reach indices
    sorted_ids = [obs_id for obs_id, _ in sorted(obs_to_reach_mapping.items(), key=lambda item: item[1])]
    df_obs = df_obs[sorted_ids].copy()
    df_obs.rename(columns=obs_to_reach_mapping, inplace=True)
    df_obs = df_obs.reset_index()
    
    
    print("Loading predicted GCN-Q data...")
    # File paths
    calibration_file = save_dir+"\Q_Calibration_Preds_seq" + str(seq_len) +".xlsx"
    validation_file  = save_dir+"\Q_Validation_Preds_seq" + str(seq_len) +".xlsx"
    
    # Read Excel files
    calibration_df = pd.read_excel(calibration_file)
    validation_df  = pd.read_excel(validation_file)
    
    # Combine both
    df_pred = pd.concat([calibration_df, validation_df], ignore_index=True)
    
        
    # Build list of station parameter sets (one DataFrame per reach)
    reach_ids = output_sub_df["Subbasin"].unique().astype(int)
    station_ps = []
    for reach_id in reach_ids:
        swat_df = output_final[output_final["Subbasin"] == reach_id].copy().reset_index(drop=True)
        # Clip to desired end date
        swat_df = swat_df[swat_df["Date"] <= SWAT_dates[1]].copy()

        # Merge observed data
        if reach_id in df_obs.columns:
            swat_df = pd.merge(swat_df, df_obs[['Date', reach_id]].rename(columns={reach_id: "Obs_DOC"}), on="Date", how="left")
        else:
            swat_df["Obs_DOC"] = np.nan
            
        
        df_pred_r = df_pred[df_pred["Subbasin"] == reach_id]
        
        if not df_pred_r.empty:
            swat_df = pd.merge(
                swat_df,
                df_pred_r[["Date", "GCN_Q"]],
                on="Date",
                how="left"
            )
        else:
            swat_df["GCN_Q"] = np.nan


        # Warmup removal
        wu_start_date = dates[0] + pd.DateOffset(years=warmup_years)
        swat_df = swat_df[swat_df["Date"] >= wu_start_date].copy().reset_index(drop=True)
        # Add surface runoff ratio only
        swat_df["Runoff_Ratio"] = swat_df["SURQmm"] / (swat_df["PRECIPmm"] + 1e-6)
        # swat_df["Baseflow_Ratio"] = swat_df["GW_Qmm"] / (swat_df["WYLDmm"].replace(0, np.nan) + 1e-6)
        swat_df["Quickflow_Ratio"] = swat_df["SURQmm"] / (swat_df["WYLDmm"].replace(0, np.nan) + 1e-6)
        # Clean invalid values
        swat_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        
        # Add temporal features
        swat_df = create_temporal_features(swat_df)

        # Feature Engineering (Lags/AWI)
        # Lags (keep for Transformer)
        for lag in range(1, 4):
            swat_df[f"PRECIPmm_lag{lag}"] = swat_df["PRECIPmm"].shift(lag)
        
        # 3-day antecedent wetness
        swat_df["PRECIP_3day_sum"] = swat_df["PRECIPmm"].rolling(3).sum()
        swat_df["AWI_3"] = (
                            0.6 * swat_df["PRECIPmm"].shift(1) +
                            0.3 * swat_df["PRECIPmm"].shift(2) +
                            0.1 * swat_df["PRECIPmm"].shift(3)
                        )

        
        # Event-based storm index (DO NOT smooth)
        swat_df["Storm_Index"] = swat_df["PRECIPmm"] / (swat_df["AWI_3"] + 1e-6)
        
        # Extreme events based on DAILY rainfall (not accumulated)
        p95 = swat_df["PRECIPmm"].quantile(0.95)
        swat_df["ExtremeStorm"] = (swat_df["PRECIPmm"] >= p95).astype(int)
        
        # FLOW routing memory
        # ─── Flow memory from GCN discharge ───
        swat_df["Q_lag1"] = swat_df["GCN_Q"].shift(1)
        swat_df["Q_lag2"] = swat_df["GCN_Q"].shift(2)
        
        # ─── dQ features (GCN-based) ───
        swat_df["dQ"]  = swat_df["GCN_Q"] - swat_df["Q_lag1"]
        swat_df["dQ2"] = swat_df["Q_lag1"] - swat_df["Q_lag2"]
        swat_df["dQ_norm"] = swat_df["dQ"] / (swat_df["Q_lag1"].abs() + 1e-6)


        
        # Drop rows consistently (VERY IMPORTANT)
        swat_df = swat_df.dropna(subset=["PRECIPmm_lag3","Q_lag2","dQ"]).reset_index(drop=True)
        
        # Remove non-DOC variables
        swat_df.drop(
            columns=[
                "WYLDmm",
                "Quickflow_Ratio",
                "month",
                "Type",
                "FLOW_INcms",
                "SURQmm"
            ],
            errors="ignore",
            inplace=True
        )

        station_ps.append(swat_df)

    return station_ps



def build_edge_index(shapefile_path: str) -> torch.Tensor:
    """Constructs the directed graph edge index (adjacency list) from the reach shapefile."""
    print("Building edge index...")

    # Find and read reach shapefile
    riv_files = [f for f in os.listdir(shapefile_path) if f.lower().startswith("riv") and f.endswith(".shp")]
    if not riv_files:
        raise FileNotFoundError("No riv shapefile found in folder")
    riv_shp_path = os.path.join(shapefile_path, riv_files[0])

    df_reach = gpd.read_file(riv_shp_path)
    df_reach = pd.DataFrame(df_reach)

    # Filter out potential NaNs or invalid final row
    df_reach = df_reach.dropna(subset=['FROM_NODE', 'TO_NODE']).reset_index(drop=True)

    from_nodes_raw = df_reach['FROM_NODE'].values.astype(int)
    to_nodes_raw = df_reach['TO_NODE'].values.astype(int)

    all_nodes = np.unique(np.concatenate([from_nodes_raw, to_nodes_raw]))
    # Map SWAT reach ID to GNN node index (0 to N-1)
    swat_to_gnn_id = {swat_id: idx for idx, swat_id in enumerate(all_nodes)}

    # Convert SWAT IDs to GNN indices
    from_nodes = np.array([swat_to_gnn_id.get(n, -1) for n in from_nodes_raw if n in swat_to_gnn_id])
    to_nodes = np.array([swat_to_gnn_id.get(n, -1) for n in to_nodes_raw if n in swat_to_gnn_id])

    # Filter out invalid nodes (-1) - though swat_to_gnn_id should handle this
    valid_mask = (from_nodes != -1) & (to_nodes != -1)
    from_nodes = from_nodes[valid_mask]
    to_nodes = to_nodes[valid_mask]

    edge_index = torch.tensor([from_nodes, to_nodes], dtype=torch.long)
    return edge_index


def clip_parameter_sets(station_ps: list[pd.DataFrame]) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[pd.DataFrame]]:
    """Clips the full list of DataFrames into calibration, validation, and testing sets."""
    calibration_sets, validation_sets, testing_sets = [], [], []
    for df in station_ps:
        df = df.copy().set_index("Date")
        calibration_sets.append(df.loc[cali_period[0]:cali_period[1]].reset_index())
        validation_sets.append(df.loc[vali_period[0]:vali_period[1]].reset_index())
        testing_sets.append(df.loc[test_period[0]:test_period[1]].reset_index())
    return calibration_sets, validation_sets, testing_sets


# ==============================================================================
## 5. Function: Predict and Evaluate
# ==============================================================================

def prepare_dataset_pipeline(
    swat_folder_path,
    sim_name,
    excel_path,
    obs_to_reach_mapping,
    SWAT_Model_Dates,
    warm_up_years,
    final_outlet,
    threshold,
    save_dir,
    device,
    k2u_node,
    seq_len
):
    """
    Prepare all datasets, graph structure, and selected features 
    for the Enhanced SWAT–GNN–Transformer pipeline.

    Returns
    -------
    cali_ps, vali_ps, test_ps : list
        Station dataframes for each period.
    edge_index : torch.Tensor
        Graph connectivity on device.
    selected_features : list
        Input features used for model training.
    converted_reaches_default : list
        Reaches hidden during training.
    importance_df : pd.DataFrame
        Feature importance table.
    """

    print("🚀 Starting enhanced pipeline with feature selection...")
    os.makedirs(save_dir, exist_ok=True)

    # ---------------------------------------
    # 1. Build Paths
    # ---------------------------------------
    swat_path = f"{swat_folder_path}/Scenarios/{sim_name}/TxtInOut"
    shapefile_path = f"{swat_folder_path}/Watershed/Shapes"

    # ---------------------------------------
    # 2. Load data & Graph Structure
    # ---------------------------------------
    station_ps = load_Parameter_Set(
        swat_path,
        excel_path,
        obs_to_reach_mapping,
        shapefile_path=shapefile_path,
        SWAT_dates=SWAT_Model_Dates,
        warmup_years=warm_up_years,
        seq_len = seq_len,
        save_dir=save_dir
    )

    edge_index = build_edge_index(shapefile_path).to(device)


    # ---------------------------------------
    # 3. Split into Calibration / Validation / Testing
    # ---------------------------------------
    cali_ps, vali_ps, test_ps = clip_parameter_sets(station_ps)

    # ---------------------------------------
    # 4. Identify real reaches (with real discharge)
    # ---------------------------------------
    real_known_reaches = [
        i for i, df in enumerate(cali_ps) 
        if df['Obs_DOC'].notna().sum() > 0
    ]

    # ---------------------------------------
    # 5. Select converted (hidden) reaches
    # ---------------------------------------
    import random
    random.seed(4)

    available_reaches = [
        r for r in real_known_reaches 
        if r != final_outlet - 1
    ]  # Exclude final outlet always

    converted_reaches_default = random.sample(
        available_reaches, 
        min(1, len(available_reaches))
    )
    
    converted_reaches_default = [k2u_node-1]
    
    
    print("🔒 Converted reaches (hidden during training):", converted_reaches_default[0]+1)

    # Save converted reaches to file
    with open(os.path.join(save_dir, "DOC_converted_reaches.txt"), "w") as f:
        f.write("\n".join(map(str, converted_reaches_default)))

    # ---------------------------------------
    # 6. Feature Selection
    # ---------------------------------------
    candidate_features = [
        'FLOW_INcms','FLOW_OUTcms','Subbasin','PRECIPmm','SNOMELTmm','PETmm','ETmm',
        'SWmm','PERCmm','SURQmm','GW_Qmm','WYLDmm','SYLDt/ha','Type','Area','Slo1',
        'Len1','Sll','Csl','Wid1','Dep1','Lat','Long_','Elev','ElevMin','ElevMax',
        'Month','day_of_year','sin_day','cos_day','year'
    ]

    # remove duplicates
    candidate_features = list(dict.fromkeys(candidate_features))

    try:
        selected_features, importance_df = select_relevant_features(
            cali_ps,
            target_col='Obs_DOC',
            importance_threshold=threshold,
            drop_candidates=['Date', 'Obs_DOC', 'year', 'day_of_year','FLOW_OUTcms','GCN_Q'],
            features_to_add = features_to_add
        )

        print("🎯 Selected Features:", selected_features)

    except Exception as e:
        print("❌ Feature selection failed:", e)

        # Fallback based on hydrological reasoning
        selected_features = [
            'PRECIPmm', 'SURQmm', 'GW_Qmm', 'SWmm',
            'PERCmm', 'FLOW_OUTcms', 'sin_day', 'cos_day',
            'Month', 'Area', 'Slo1', 'Len1'
        ]

        importance_df = pd.DataFrame({
            'Feature': selected_features,
            'Importance': [1/len(selected_features)] * len(selected_features)
        })

        importance_df.to_excel(
            os.path.join(save_dir, "DOC_Feature_Importance_RF_fallback.xlsx"),
            index=False
        )

        print("⚠ Using fallback selected_features:", selected_features)

    # Save selected features
    with open(os.path.join(save_dir, "DOC_selected_features.txt"), "w") as f:
        f.write("\n".join(selected_features))

    return (
        cali_ps,
        vali_ps,
        test_ps,
        edge_index,
        selected_features,
        converted_reaches_default,
        importance_df
    )




def train_model_for_seq_len(
    seq_len,
    cali_ps,
    vali_ps,
    selected_features,
    edge_index,
    converted_reaches_default,
    device,
    save_dir,
    ImprovedTransformerGCN,
    criterion_class,
    hidden_trans,
    hidden_gcn,
    dropout_rate,
    num_heads,
    trans_layers,
    learning_rate,
    weight_decay,
    epochs,
    early_stop_patience,
    min_delta,
    transformations,
    LR_method,
    plot_loss=True

):
    """
    Train a Transformer-GCN model for a given sequence length with early stopping.

    Parameters
    ----------
    seq_len : int
        Sequence length for temporal features.
    cali_ps : list
        Calibration period station DataFrames.
    vali_ps : list
        Validation period station DataFrames.
    selected_features : list
        Features used for training.
    edge_index : torch.Tensor
        Graph edge connections.
    converted_reaches_default : list
        Indices to exclude from training.
    device : torch.device
        Target device ('cuda' or 'cpu').
    save_dir : str
        Directory to save model and plots.
    ImprovedTransformerGCN : nn.Module
        Model class definition.
    criterion_class : nn.Module
        Loss function class (e.g., KGELoss, HydroGraphLoss).
    hidden_trans, hidden_gcn, dropout_rate, num_heads, trans_layers, learning_rate, weight_decay : hyperparameters
    epochs, early_stop_patience, min_delta : training control parameters.
    plot_loss : bool, default=True
        Whether to plot training/validation loss history.

    Returns
    -------
    best_model : nn.Module
        Trained model with best validation performance.
    scaler : sklearn.StandardScaler
        Scaler fitted on training data.
    train_losses, val_losses : list
        Training and validation loss histories.
    """

    print(f"\n========================================\nRunning seq_len={seq_len}\n========================================")

    # ─── 1. PREPARE TRAINING DATA ───
    create_lag_features1 = create_lag_features(cali_ps, seq_len, selected_features=selected_features)
    
    if transformations=='Original':
        features_list, targets_list, mask_list = create_lag_features1["original"]
    elif transformations=='Log':
        features_list, targets_list, mask_list = create_lag_features1["log"]

    # Mask out converted reaches
    mask_list_train = [m.copy() for m in mask_list]
    for idx in converted_reaches_default:
        if idx < len(mask_list_train):
            mask_list_train[idx] = np.zeros_like(mask_list_train[idx], dtype=bool)

    X_stacked = np.vstack([
        f.reshape(f.shape[0], -1) if f.size else np.zeros((0, len(selected_features) * seq_len))
        for f in features_list
    ])
    
    X_stacked = np.nan_to_num(X_stacked, nan=0.0, posinf=0.0, neginf=0.0) ## Added

    y_stacked = np.concatenate([t if t.size else np.zeros((0,)) for t in targets_list])
    mask_stacked = np.concatenate([m if m.size else np.zeros((0,), dtype=bool) for m in mask_list_train])

    if X_stacked.shape[0] == 0:
        print(f"No training samples for seq_len={seq_len}, skipping...")
        return None, None, None, None

    # Standardize and convert to tensors
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_stacked)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float).to(device)
    y_tensor = torch.tensor(y_stacked, dtype=torch.float).to(device)
    mask_tensor = torch.tensor(mask_stacked, dtype=torch.bool).to(device)

    data = Data(x=X_tensor, edge_index=edge_index, y=y_tensor)
    data.train_mask = mask_tensor
    data = data.to(device)

    # ─── 2. PREPARE VALIDATION DATA ───
    create_lag_features1 = create_lag_features(vali_ps, seq_len, selected_features=selected_features)

    if transformations=='Original':
        features_list_val, targets_list_val, mask_list_val = create_lag_features1["original"]
    elif transformations=='Log':
        features_list_val, targets_list_val, mask_list_val = create_lag_features1["log"]
   

    X_val = np.vstack([
        f.reshape(f.shape[0], -1) if f.size else np.zeros((0, len(selected_features) * seq_len))
        for f in features_list_val
    ])
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0) ## Added
    y_val = np.concatenate([t if t.size else np.zeros((0,)) for t in targets_list_val])
    mask_val = np.concatenate([m if m.size else np.zeros((0,), dtype=bool) for m in mask_list_val])

    if X_val.shape[0] == 0:
        print("⚠️ Warning: No validation samples available.")
        X_val_scaled = np.zeros_like(X_scaled)
    else:
        X_val_scaled = scaler.transform(X_val)

    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float).to(device)
    mask_val_tensor = torch.tensor(mask_val, dtype=torch.bool).to(device)

    data_val = Data(x=X_val_tensor, edge_index=edge_index, y=y_val_tensor)
    data_val.train_mask = mask_val_tensor
    data_val = data_val.to(device)

    # ─── 3. MODEL INIT ───
    n_features_per_time = X_scaled.shape[1] // seq_len
    current_num_heads = max(1, next((h for h in range(num_heads, 0, -1) if n_features_per_time % h == 0), 1))

    model = ImprovedTransformerGCN(
        n_features=n_features_per_time,
        hidden_trans=hidden_trans,
        hidden_gcn=hidden_gcn,
        seq_len=seq_len,
        dropout=dropout_rate,
        num_heads=current_num_heads,
        trans_layers=trans_layers
    ).to(device)
    
    if LR_method == "ReduceLROnPlateau":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5)
        
    elif LR_method == "Cosine":
        optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.0
    )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=200,      # first cycle length (epochs)
            T_mult=1,     # keep cycle length fixed
            eta_min=0.00005
        )
    elif LR_method == "OneCycle":
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate
        )
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.0008,
            epochs=epochs,
            steps_per_epoch=1,
            pct_start=0.3,
            anneal_strategy="cos",
            final_div_factor=10
        )

   
    
    loss_registry = {
        "MSE": nn.MSELoss,
        "HydroGraphLoss": HydroGraphLoss,
        "KGELoss": KGELoss,
        "DifferentialHydroLoss": DifferentialHydroLoss
        
        # Add more losses here
    }
    criterion_class = loss_registry[Loss]
    criterion = criterion_class().to(device)

    best_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    # ─── 4. TRAINING LOOP ───
    pbar = tqdm(range(epochs), desc=f"Seq Len {seq_len} Training", unit="epoch", colour="red")

    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

        out = model(data)
        train_idx = data.train_mask
        if train_idx.sum() == 0:
            print("No training indices. Breaking.")
            break

        loss = criterion(out[train_idx], data.y[train_idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step() ##*

        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(data_val)
            val_idx = data_val.train_mask
            val_loss = criterion(val_out[val_idx], data_val.y[val_idx]) if val_idx.sum() > 0 else torch.tensor(float('inf'))

        # ─── Correct scheduler stepping ───
        if LR_method == "ReduceLROnPlateau":
            scheduler.step(val_loss)
        
        elif LR_method == "Cosine":
            scheduler.step(epoch)
        
        elif LR_method == "OneCycle":
            scheduler.step()

        train_losses.append(loss.item())
        val_losses.append(val_loss.item() if hasattr(val_loss, "item") else float("inf"))

        # Early Stopping
        current_val_loss = val_losses[-1]
        if current_val_loss < best_loss - min_delta:
            best_loss = current_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, f"DOC_best_model_seq{seq_len}.pt"))
            patience_counter = 0
        else:
            patience_counter += 1

        pbar.set_postfix(
            Best_Validation_Loss=f"{best_loss:.4f}",
            Training_Loss=f"{loss.item():.4f}",
            Validation_Loss=f"{current_val_loss:.4f}",
            Patience=f"{patience_counter}/{early_stop_patience}"
        )

        if patience_counter >= early_stop_patience:
            print(f"\n🛑 Early stopping at epoch {epoch+1}. Best Val Loss: {best_loss:.6f}")
            break

    # ─── 5. LOAD BEST MODEL ───
    best_model = ImprovedTransformerGCN(
        n_features=n_features_per_time,
        hidden_trans=hidden_trans,
        hidden_gcn=hidden_gcn,
        seq_len=seq_len,
        dropout=dropout_rate,
        num_heads=current_num_heads,
        trans_layers=trans_layers
    ).to(device)
    

    best_model.load_state_dict(torch.load(os.path.join(save_dir, f"DOC_best_model_seq{seq_len}.pt")))
    best_model.eval()

    # ─── 6. PLOT TRAINING HISTORY ───
    if plot_loss:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training History - seq_len={seq_len}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"DOC_training_history_seq{seq_len}.png"), dpi=300)
        plt.close()

    print(f"✅ Training completed for seq_len={seq_len} | Best Validation Loss = {best_loss:.4f}")
    # ─────────────────────────────────────────────────────────────
    # 7. SAVE TRAINING (CALIBRATION) PREDICTIONS — SAFE & ALIGNED
    # ─────────────────────────────────────────────────────────────
    print("\n💾 Saving TRAINING predictions (Calibration)...")
    
    best_model.eval()
    with torch.no_grad():
        preds = best_model(data).cpu().numpy()
        obs = y_tensor.cpu().numpy()
    
    # --- Split predictions back per subbasin ---
    lengths_per_node = [f.shape[0] for f in features_list]
    
    split_preds = []
    idx = 0
    for l in lengths_per_node:
        if l == 0:
            split_preds.append(np.array([]))
        else:
            split_preds.append(preds[idx:idx + l])
            idx += l
    
    df_with_preds = []
    
    for i, df_station in enumerate(cali_ps):
        df_copy = df_station.copy().reset_index(drop=True)
    
        preds_i = split_preds[i]
        pad_len = len(df_copy) - len(preds_i)
    
        # pad initial seq_len with NaN
        preds_i_full = np.concatenate([np.full(pad_len, np.nan), preds_i])
    
        # inverse log if needed
        if transformations == "Log":
            preds_i_full = inverse_log_transform(preds_i_full)
    
        preds_i_full = np.clip(preds_i_full, 0.0, None)
    
        df_copy["GCN_DOC"] = preds_i_full
    
        # ensure Date exists
        if "Date" not in df_copy.columns or df_copy["Date"].isnull().all():
            df_copy["Date"] = pd.date_range(
                start=cali_ps[0]["Date"].iloc[0],
                periods=len(df_copy),
                freq="D"
            )
    
        df_with_preds.append(df_copy)
    
    df_preds_train = pd.concat(df_with_preds, ignore_index=True)
    
    # --- Save to Excel ---
    train_pred_file = os.path.join(save_dir, f"DOC_Calibration_Preds_seq{seq_len}.xlsx")
    df_preds_train.to_excel(train_pred_file, index=False)
    
    print(f"✅ Calibration predictions saved to: {train_pred_file}")

    
    # ─────────────────────────────────────────────────────────────
    # Metric computation (IDENTICAL to validation)
    # ─────────────────────────────────────────────────────────────

    metrics_list = []
    phase_name = "Calibration"
    
    for subbasin in df_preds_train['Subbasin'].unique():
        df_sub = df_preds_train[df_preds_train['Subbasin'] == subbasin].copy()
        mask_valid = df_sub['Obs_DOC'].notna() & df_sub['GCN_DOC'].notna()
        df_valid = df_sub[mask_valid]
    
        if len(df_valid) == 0:
            metrics = dict(
                KGE_GCN=np.nan, KGE_SWAT=np.nan,
                NSE_GCN=np.nan, NSE_SWAT=np.nan,
                R2_GCN=np.nan,  R2_SWAT=np.nan,
                PBIAS_GCN=np.nan, PBIAS_SWAT=np.nan
            )
        else:
            obs_vals  = df_valid['Obs_DOC'].values
            pred_vals = df_valid['GCN_DOC'].values
            swat_vals = df_valid['FLOW_OUTcms'].values
    
            # ─── Plot Hydrograph & Scatter ───
            sub_dir = os.path.join(save_dir, f"subbasin_{subbasin}")
            os.makedirs(sub_dir, exist_ok=True)
    
            try:
                results_df = pd.DataFrame({
                    "Date": df_valid['Date'],
                    "Obs(kg/day)": df_valid['Obs_DOC'],
                    "Pred(kg/day)": df_valid['GCN_DOC'],
                    "PRECIPmm": df_valid['PRECIPmm']
                })
    
                plot_hydrograph(
                    results_df,
                    save_dir=sub_dir,
                    station_id=f"subbasin_{subbasin}_seq{seq_len}_{phase_name.lower()}"
                )
    
                plot_scatter(
                    obs_vals,
                    pred_vals,
                    f"Subbasin {subbasin} - Scatter ({phase_name})",
                    os.path.join(
                        sub_dir,
                        f"DOC_scatter_seq{seq_len}_{phase_name.lower()}.png"
                    )
                )
    
            except Exception as e:
                print(f"⚠️ Plotting failed for subbasin {subbasin} ({phase_name}): {e}")
    
            # ─── Metric Computation ───
            def safe_eval(func, *args):
                try:
                    if func.__name__ == "kge":
                        return float(func(*args)[0])
                    return float(func(*args))
                except Exception:
                    return np.nan
    
            metrics = {
                "KGE_GCN": safe_eval(kge, pred_vals, obs_vals),
                "KGE_SWAT": safe_eval(kge, swat_vals, obs_vals),
                "NSE_GCN": safe_eval(nse, pred_vals, obs_vals),
                "NSE_SWAT": safe_eval(nse, swat_vals, obs_vals),
                "R2_GCN": safe_eval(r2_score, obs_vals, pred_vals),
                "R2_SWAT": safe_eval(r2_score, obs_vals, swat_vals),
                "PBIAS_GCN": safe_eval(pbias, pred_vals, obs_vals),
                "PBIAS_SWAT": safe_eval(pbias, swat_vals, obs_vals),
            }
    
        metrics.update({
            "Subbasin": subbasin,
            "seq_len": seq_len,
            "Period": phase_name
        })
        metrics_list.append(metrics)
    
    # ─── Save Metrics ───
    metrics_df = pd.DataFrame(metrics_list)
    metrics_file = os.path.join(save_dir, f"DOC_{phase_name}_Metrics_seq{seq_len}.xlsx")
    metrics_df.to_excel(metrics_file, index=False)
    
    print(f"\n📊 Metrics for {phase_name} | seq_len={seq_len}")
    print(metrics_df)


    return best_model, scaler, train_losses, val_losses


def predict_and_evaluate_phase(
    model, phase_name, seq_len, data_list, period, save_dir,
    scaler, edge_index, selected_features, device,
    plot_hydrograph, plot_scatter, transformations
):
    """
    Common function for Validation and Testing prediction & evaluation.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model for inference.
    phase_name : str
        Either "Validation" or "Testing".
    seq_len : int
        Sequence length used for temporal input.
    data_list : list
        List of per-station DataFrames (validation or testing).
    period : tuple
        Start and end date (used if Date missing).
    save_dir : str
        Directory to save outputs.
    scaler : sklearn scaler
        Fitted scaler for input features.
    edge_index : torch.tensor
        Graph edges.
    selected_features : list
        Features used in training.
    device : torch.device
        Torch device.
    plot_hydrograph, plot_scatter : functions
        Plotting utilities.
    """

    print(f"\n--- {phase_name} for seq_len={seq_len} ---")

    # ─── Create Lag Features ───
    create_lag_features1 = create_lag_features(
        data_list, seq_len, selected_features=selected_features
    )

    
    if transformations=='Original':
        features_list, targets_list, mask_list = create_lag_features1["original"]
    elif transformations=='Log':
        features_list, targets_list, mask_list = create_lag_features1["log"]
    

    X = np.vstack([
        f.reshape(f.shape[0], -1) if f.size else np.zeros((0, len(selected_features) * seq_len))
        for f in features_list
    ])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0) ## Added
    y = np.concatenate([t if t.size else np.zeros((0,)) for t in targets_list])
    mask = np.concatenate([m if m.size else np.zeros((0,), dtype=bool) for m in mask_list])

    # ─── Handle Empty Case ───
    if X.shape[0] == 0:
        print(f"No {phase_name.lower()} samples available for seq_len={seq_len}")
        return None, None

    # ─── Scale Inputs and Convert to Torch ───
    X_scaled = scaler.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float).to(device)
    mask_tensor = torch.tensor(mask, dtype=torch.bool).to(device)

    data = Data(x=X_tensor, edge_index=edge_index, y=y_tensor)
    data.train_mask = mask_tensor
    data = data.to(device)

    # ─── Prediction ───
    with torch.no_grad():
        preds = model(data).cpu().numpy()
        obs = y_tensor.cpu().numpy()

    # ─── Reshape and Align Predictions ───
    lengths_per_node = [f.shape[0] for f in features_list]
    split_preds = []
    idx = 0
    for l in lengths_per_node:
        if l == 0:
            split_preds.append(np.array([]))
        else:
            split_preds.append(preds[idx:idx + l])
            idx += l

    df_with_preds = []
    for i, df in enumerate(data_list):
        df_copy = df.copy().reset_index(drop=True)
        preds_i = split_preds[i]
        pad_len = len(df_copy) - len(preds_i)
        preds_i_full = np.concatenate([np.full(pad_len, np.nan), preds_i])
        
        if transformations=='Log':
            # Inverse log transform and ensure non-negative
            with np.errstate(over='ignore'):
                preds_i_full = inverse_log_transform(preds_i_full)

        preds_i_full = np.where(preds_i_full < 0, 0.0, preds_i_full)  # No negative discharge
        df_copy["GCN_DOC"] = preds_i_full

        if 'Date' not in df_copy.columns or df_copy['Date'].isnull().all():
            df_copy["Date"] = pd.date_range(start=period[0], periods=len(df_copy), freq="D")

        df_with_preds.append(df_copy)

    df_preds = pd.concat(df_with_preds, ignore_index=True)
    output_file = os.path.join(save_dir, f"DOC_{phase_name}_Preds_seq{seq_len}.xlsx")
    df_preds.to_excel(output_file, index=False)
    print(f"{phase_name} predictions saved to: {output_file}")

    # ─── Compute Metrics ───
    metrics_list = []
    for subbasin in df_preds['Subbasin'].unique():
        df_sub = df_preds[df_preds['Subbasin'] == subbasin].copy()
        mask_valid = df_sub['Obs_DOC'].notna() & df_sub['GCN_DOC'].notna()
        df_valid = df_sub[mask_valid]

        if len(df_valid) == 0:
            metrics = dict(KGE_GCN=np.nan, KGE_SWAT=np.nan,
                           NSE_GCN=np.nan, NSE_SWAT=np.nan,
                           R2_GCN=np.nan, R2_SWAT=np.nan,
                           PBIAS_GCN=np.nan, PBIAS_SWAT=np.nan)
        else:
            obs_vals = df_valid['Obs_DOC'].values
            pred_vals = df_valid['GCN_DOC'].values
            swat_vals = df_valid['FLOW_OUTcms'].values

            # Plot hydrograph and scatter
            sub_dir = os.path.join(save_dir, f"subbasin_{subbasin}")
            os.makedirs(sub_dir, exist_ok=True)
            try:
                results_df = pd.DataFrame({
                    "Date": df_valid['Date'],
                    "Obs(kg/day)": df_valid['Obs_DOC'],
                    "Pred(kg/day)": df_valid['GCN_DOC'],
                    "PRECIPmm": df_valid['PRECIPmm']
                })
                plot_hydrograph(results_df, save_dir=sub_dir, station_id=f"subbasin_{subbasin}_seq{seq_len}_{phase_name.lower()}")
                plot_scatter(obs_vals, pred_vals, f"Subbasin {subbasin} - Scatter Plot ({phase_name})",
                             os.path.join(sub_dir, f"DOC_scatter_seq{seq_len}_{phase_name.lower()}.png"))
            except Exception as e:
                print(f"Plotting failed for subbasin {subbasin} ({phase_name}): {e}")

            # Compute Metrics
            def safe_eval(func, *args):
                try:
                    if func.__name__ == "kge":
                        return float(func(*args)[0])  # extract the KGE value
                    return float(func(*args))
                except Exception:
                    return np.nan

            metrics = {
                "KGE_GCN": safe_eval(kge, pred_vals, obs_vals),
                "KGE_SWAT": safe_eval(kge, swat_vals, obs_vals),
                "NSE_GCN": safe_eval(nse, pred_vals, obs_vals),
                "NSE_SWAT": safe_eval(nse, swat_vals, obs_vals),
                "R2_GCN": safe_eval(r2_score, obs_vals, pred_vals),
                "R2_SWAT": safe_eval(r2_score, obs_vals, swat_vals),
                "PBIAS_GCN": safe_eval(pbias, pred_vals, obs_vals),
                "PBIAS_SWAT": safe_eval(pbias, swat_vals, obs_vals)
            }

        metrics.update({"Subbasin": subbasin, "seq_len": seq_len, "Period": phase_name})
        metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list)
    metrics_file = os.path.join(save_dir, f"DOC_{phase_name}_Metrics_seq{seq_len}.xlsx")
    metrics_df.to_excel(metrics_file, index=False)
    print(f"Metrics for {phase_name} seq_len={seq_len}:")
    print(metrics_df)

    return df_preds, metrics_df