# ==============================================================================
## Imports & Configuration
# ==============================================================================


# ───────── 🧩 Model Hyperparameters ─────────
hidden_gcn      = 8        # Number of hidden units in the GCN layer (spatial feature learning)
hidden_trans    = 8        # Number of hidden units in the Transformer (temporal feature learning)
num_heads       = 4        # Number of attention heads in the Transformer
trans_layers    = 2        # Number of Transformer encoder layers
dropout_rate    = 0.1      # Dropout probability used for regularization

seq_len_list    = [7]      # Length of input time sequence (days) used for temporal modeling
threshold       = 0.01     # Small threshold used for numerical stability in loss calculations
Loss            = "KGELoss"  # Loss function used for training (KGELoss, HydroGraphLoss, MSE, HydroCompositeLoss)
features_to_add = []       # Optional additional features to include in model inputs


# ───────── 🏋️ Training Hyperparameters ─────────
epochs              = 50     # Maximum number of training epochs
early_stop_patience = 5      # Number of epochs to wait before early stopping if no improvement
learning_rate       = 5e-4   # Learning rate for the optimizer
weight_decay        = 0.0    # L2 regularization strength applied to model weights
min_delta           = 1e-4   # Minimum improvement required to reset early stopping

LR_method           = "Cosine"   # Learning rate scheduling method (ReduceLROnPlateau, Cosine, OneCycle)


# ───────── 📆 Data Periods ─────────
SWAT_Model_Dates = ["2014-01-01", "2019-12-31"]  # Full SWAT simulation period used for data extraction

cali_period = ["2014-01-01", "2017-12-31"]       # Model calibration (training) period
vali_period = ["2019-01-01", "2019-12-31"]       # Model validation period used for hyperparameter evaluation
test_period = ["2019-01-01", "2019-12-31"]       # Final testing period for performance reporting

warm_up_years = 0                                # Number of warm-up years removed from the start of SWAT simulations


# ───────── 🔗 Gauge → Reach Mapping ─────────
obs_to_reach_mapping = {
    1:1, 2:2, 3:3, 4:4, 5:5,
    6:6, 7:7, 8:8, 9:9, 10:10,
    11:11, 12:12, 13:13, 14:14, 15:15,
    16:16, 17:17, 18:18, 19:19
}
# Dictionary mapping observed gauge station IDs to SWAT reach IDs

final_outlet = 19   # Reach ID representing the watershed outlet
k2u_node     = 5    # Node used for special routing or graph structure adjustments


# ───────── 🔄 Data Transformation ─────────
transformations = "Original"   # Data transformation applied to target variables ("Original" or "Log")


# ───────── 📁 Input Files ─────────
excel_path = r"C:\Users\venky\Dropbox\GIST\Work\Projects\SWAT_GAT_Fusion\Results\TCW_OBS.xlsx"
# Path to Excel file containing observed streamflow or water quality measurements

swat_folder_path = r"D:\GIST\Projects\Fusion_WQ\TCW_data\SWAT file"
# Path to SWAT model directory containing watershed configuration and simulation outputs

sim_name = "Default"
# SWAT scenario name used to extract simulation outputs


# ───────── 💾 Output Directory ─────────
project_root = r"C:\Users\venky\Dropbox\GIST\Work\Projects\SWAT_GAT_Fusion\Manuscript\Git_Hub\Check"
# Root directory where model outputs, results, and figures will be saved