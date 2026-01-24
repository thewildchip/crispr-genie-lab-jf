from pathlib import Path

# Base folders
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # points to CRISPR-GENIE-LAB/

# Data paths
DATA_RAW = BASE_DIR / "data/raw"
DATA_PROCESSED = BASE_DIR / "data/processed"
DATA_PROCESSED_EFFICACY = DATA_PROCESSED / "efficacy"
DATA_PROCESSED_OFFTARGET = DATA_PROCESSED / "off_target"
DATA_PROCESSED_OFFTARGET_OLD = DATA_PROCESSED / "off_target_old"

# Model paths
MODELS_EFFICACY = BASE_DIR / "models/efficacy"
MODELS_OFFTARGET = BASE_DIR / "models/off_target"
MODELS_OFFTARGET_OLD = BASE_DIR / "models/off_target_old"

# Notebook paths (optional)
NOTEBOOKS_EFFICACY = BASE_DIR / "notebooks/efficacy"
NOTEBOOKS_OFFTARGET = BASE_DIR / "notebooks/off_target"
