from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "maven"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
TRAIN_BATCH_SIZE = 8


EVAL_BATCH_SIZE = 8
NUM_EPOCHS = 3
RANDOM_SEED = 42
