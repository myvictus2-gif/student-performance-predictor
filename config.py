"""Project settings in one place: file paths, column names and valid ranges."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
for d in (MODELS_DIR, REPORTS_DIR):
    d.mkdir(exist_ok=True)

SYNTHETIC_FILE = DATA_DIR / "student_data.csv"
KAGGLE_FILE = DATA_DIR / "StudentPerformanceFactors.csv"   # optional, used if present

SEED = 42
FEATURES = ["study_hours", "attendance", "sleep_hours", "previous_marks"]
TARGET = "final_marks"

# Plain-English names used in charts and the app
LABELS = {
    "study_hours": "Study hours",
    "attendance": "Attendance (%)",
    "sleep_hours": "Sleep hours",
    "previous_marks": "Previous exam marks",
    "final_marks": "Final marks",
}

# Values outside these ranges are impossible and are treated as errors.
VALID_RANGES = {
    "study_hours": (0, 24),      # per day for the synthetic data
    "attendance": (0, 100),
    "sleep_hours": (2, 14),
    "previous_marks": (0, 100),
    "final_marks": (0, 100),
}

# "Accuracy" for a regression model = share of predictions within this many marks.
ACCURACY_TOLERANCE = 5
