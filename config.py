import os
import torch
import multiprocessing

# Dados selecionados
ONE_CLASS = "spam"  # "spam" ou "phishing"
assert ONE_CLASS in [
    "spam",
    "phishing",
], f"ONE_CLASS deve ser 'spam' ou 'phishing'. Valor fornecido: {ONE_CLASS}"

NUM_WORKERS = min(8, multiprocessing.cpu_count())

# Configurações do modelo MLP e GAN
GAN_BATCH_SIZE = 512 if ONE_CLASS == "spam" else 64
MLP_AUGMENTED_BATCH_SIZE = 2048 if ONE_CLASS == "spam" else 1024
MLP_ORIGINAL_BATCH_SIZE = 1024 if ONE_CLASS == "spam" else 512
TRAIN_SPLIT = 0.75
MOVING_AVERAGE_WINDOW = 100

# GAN
NUM_EPOCHS_GAN = 3000 if ONE_CLASS == "spam" else 3000
LR_GAN = [0.0002, 0.0002]
LATENT_DIM = 1024 if ONE_CLASS == "spam" else 512
N_CRITIC = 5
LAMBDA_GP = 10
INIT_TEMPERATURE = 1.0
MIN_TEMPERATURE = 0.1
DECAY_FACTOR = 0.95
WARMUP_EPOCHS = 5
GEN_TEMPERATURE = 0.4 if ONE_CLASS == "spam" else 1.0

# PyTorch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logging
DELIMITER = "=" * 75
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# Ray Tune
NUM_SAMPLES = 20

# Caminhos
__class_name = "ham" if ONE_CLASS == "spam" else "phishing"
CHECKPOINT_DIR = "checkpoints/" + __class_name + "/"
CLASS_PATH = "_ham_" + ONE_CLASS
CACHE_DIR = "cache/"
EMAIL_CACHE_PATH = CACHE_DIR + "email_features" + CLASS_PATH + ".pkl"

# Datasets
DATASET_DIR = "Dataset/"
INDEX_PATH = DATASET_DIR + "index"
EXAMPLE_PATH = DATASET_DIR + "exampleIndex"
DATA_DIR = DATASET_DIR + "data"
SPAM_HAM_DATA_DIR = DATASET_DIR + "SpamHam/trec07p/data"
SPAM_HAM_INDEX_PATH = DATASET_DIR + "SpamHam/trec07p/full/index"
PHISHING_DIR = DATASET_DIR + "Phishing/TXTs"

# Plots
PLOT_DIR = "plots/"
MLP_ORIGINAL_PLOT_PATH = PLOT_DIR + "mlp_original" + CLASS_PATH + ".svg"
MLP_AUGMENTED_PLOT_PATH = PLOT_DIR + "mlp_augmented" + CLASS_PATH + ".svg"
GAN_PLOT_PATH = PLOT_DIR + "gan" + CLASS_PATH + ".svg"
RAYTUNE_PLOT_PATH = PLOT_DIR + "raytune_results" + CLASS_PATH + ".svg"
FD_ORIGINAL_DATA_PLOT_PATH = PLOT_DIR + "fd_original" + CLASS_PATH + ".svg"
FD_AUGMENTED_DATA_PLOT_PATH = PLOT_DIR + "fd_augmented" + CLASS_PATH + ".svg"
LR_ORIGINAL_DATA_PLOT_PATH = PLOT_DIR + "lr_original" + CLASS_PATH + ".svg"
LR_AUGMENTED_DATA_PLOT_PATH = PLOT_DIR + "lr_augmented" + CLASS_PATH + ".svg"

# Features e informações do cabeçalho
FEATURES = [
    "time_zone",
    "date_comp_date_received",
    "missing_mime-version",
    "missing_x-mailer",
    "missing_list-unsubscribe",
    "missing_x-mailman-version",
    "missing_references",
    "missing_user-agent",
    "missing_received-spf",
    "missing_x-original-to",
    "missing_domainkey-signature",
    "missing_importance",
    "missing_dmarc",
    "str_content-encoding_empty",
    "str_from_question",
    "str_from_exclam",
    "str_from_chevron",
    "str_to_chevron",
    "str_to_undisclosed",
    "str_to_empty",
    "str_message-ID_dollar",
    "str_return-path_bounce",
    "str_return-path_empty",
    "str_reply-to_question",
    "str_received-SPF_bad",
    "str_received-SPF_softfail",
    "str_received-SPF_fail",
    "str_content-type_texthtml",
    "str_precedence_list",
    "str_dmarc_bad",
    "str_dmarc_softfail",
    "str_dmarc_fail",
    "str_dkim_bad",
    "str_dkim_softfail",
    "str_dkim_fail",
    "received_str_forged",
    "email_match_from_reply-to",
    "domain_val_message-id",
    "domain_match_message-id_from",
    "domain_match_from_return-path",
    "domain_match_message-id_return-path",
    "domain_match_message-id_sender",
    "domain_match_message-id_reply-to",
    "domain_match_return-path_reply-to",
    "domain_match_reply-to_to",
    "domain_match_to_in-reply-to",
    "domain_match_errors-to_message-id",
    "domain_match_errors-to_from",
    "domain_match_errors-to_sender",
    "domain_match_errors-to_reply-to",
    "domain_match_sender_from",
    "domain_match_references_reply-to",
    "domain_match_references_in-reply-to",
    "domain_match_references_to",
    "domain_match_from_reply-to",
    "domain_match_to_from",
    "domain_match_to_message-id",
    "domain_match_to_received",
    "domain_match_reply-to_received",
    "domain_match_from_received",
    "domain_match_return-path_received",
]

HEADER_INFORMATION = [
    "label",
    "received_hop_1",
    "received_hop_2",
    "received_hop_3",
    "received_hop_4",
    "received_hop_5",
    "received_hop_6",
    "received_hop_7",
    "received_hop_8",
    "received_hop_9",
    "received_hop_10",
    "received_hop_11",
    "received_hop_12",
    "received_hop_13",
    "received_hop_14",
    "received_hop_15",
    "received_hop_16",
    "from",
    "date",
    "hop_count",
    "subject",
    "message-id",
    "to",
    "content-type",
    "mime-version",
    "x-mailer",
    "content-transfer-encoding",
    "x-mimeole",
    "x-priority",
    "return-path",
    "list-id",
    "lines",
    "x-virus-scanned",
    "status",
    "content-length",
    "precedence",
    "delivered-to",
    "list-unsubscribe",
    "list-subscribe",
    "list-post",
    "list-help",
    "x-msmail-priority",
    "x-spam-status",
    "sender",
    "errors-to",
    "reply-to",
    "x-beenthere",
    "list-archive",
    "x-mailman-version",
    "x-miltered",
    "x-uuid",
    "x-virus-status",
    "x-spam-level",
    "x-spam-checker-version",
    "references",
    "user-agent",
    "received-spf",
    "in-reply-to",
    "x-original-to",
    "user-agent",
    "arc-message-signature",
    "arc-authentication-results",
    "arc-seal",
    "thread-index",
    "cc",
    "content-disposition",
    "mailing-list",
    "x-spam-check-by",
    "domainkey-signature",
    "dkim-signature",
    "importance",
    "x-mailing-list",
]

def setup_directories() -> None:
    """Sets up the necessary directories for data and index files."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs('checkpoints/ham/', exist_ok=True)
    os.makedirs('checkpoints/phishing/', exist_ok=True)
    os.makedirs('Dataset/Phishing/TXTs/', exist_ok=True)
    os.makedirs('Dataset/SpamHam/', exist_ok=True)

if __name__ == "__main__":
    setup_directories()