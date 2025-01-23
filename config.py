import torch

# PyTorch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logging
DELIMITER = "=" * 75
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s\n"

# Caminhos
CHECKPOINT_DIR = "checkpoints/phishing/"
INDEX_PATH = "Dataset/index"
EXAMPLE_PATH = "Dataset/exampleIndex"

DATA_DIR = "Dataset/data"
SPAM_HAM_DATA_DIR = "Dataset/SpamHam/trec07p/data"
SPAM_HAM_INDEX_PATH = "Dataset/index"
PHISHING_DIR = "Dataset/Phishing/TXTs"
