import os

# Base project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths for raw data (Excel files)
RAW_DATA_PATH = os.path.join(BASE_DIR, "../data/")

# SQLite Database Path
DATABASE_PATH = os.path.join(BASE_DIR, "../database/real_estate.db")


# Raw Data Table Name
RAW_TABLE = "raw_table"
PROCESSED_TABLE = RAW_TABLE

# Predictions Table Name
PREDICTIONS_TABLE = "predictions"

# model evaluation
EVALUATION_TABLE = "grid_search_results"

# Logging Configuration
LOGGING_LEVEL = "INFO"

# Saved Models
MODELS_PATH = "../models/"