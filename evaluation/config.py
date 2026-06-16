# evaluation/config.py

from pathlib import Path

EVALUATION_DIR = Path(__file__).parent

QUESTIONS_FILE = EVALUATION_DIR / "evaluation_questions.csv"

RESULTS_FILE = EVALUATION_DIR / "evaluation_results.csv"
