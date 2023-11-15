# Data
KEY_SRC = "src"
KEY_TGT = "tgt"
KEY_EDIT = "edit"
DELIMITER_M2 = "|||"
EDIT_NONE_TYPE = {
    "noop",  # eng
    "NA",  # zho
}
EDIT_NONE_CORRECTION = {"-NONE-"}

# EditMetric
KEY_TP = "tp"
KEY_TN = "tn"
KEY_FP = "fp"
KEY_FN = "fn"

KEY_TP_EDIT = "tp_edit"
KEY_TN_EDIT = "tn_edit"
KEY_FP_EDIT = "fp_edit"
KEY_FN_EDIT = "fn_edit"

KEY_P = "P"
KEY_R = "R"
KEY_F = "F"
KEY_ACC = "Acc"

# No Length Weighting
CONFIG_NO_LW = {
    "tp": {"alpha": 1.0, "min_value": 1.0, "max_value": 1.0, "reverse": False},
    "fp": {"alpha": 1.0, "min_value": 1.0, "max_value": 1.0, "reverse": True},
    "fn": {"alpha": 1.0, "min_value": 1.0, "max_value": 1.0, "reverse": False},
}
# Default parameters for CLEME_dependent
DEFAULT_CONFIG_CORPUS_DEPENDENT = {
    "tp": {"alpha": 2.0, "min_value": 0.75, "max_value": 1.25, "reverse": False},
    "fp": {"alpha": 2.0, "min_value": 0.75, "max_value": 1.25, "reverse": True},
    "fn": {"alpha": 2.0, "min_value": 0.75, "max_value": 1.25, "reverse": False},
}
# Default parameters for CLEME_independent
DEFAULT_CONFIG_CORPUS_INDEPENDENT = {
    "tp": {"alpha": 2.0, "min_value": 0.75, "max_value": 1.25, "reverse": False},
    "fp": {"alpha": 2.0, "min_value": 0.75, "max_value": 1.25, "reverse": True},
    "fn": {"alpha": 2.0, "min_value": 0.75, "max_value": 1.25, "reverse": False},
}
# Default parameters for SentCLEME_dependent
DEFAULT_CONFIG_SENTENCE_DEPENDENT = {
    "tp": {"alpha": 10.0, "min_value": 1.00, "max_value": 10.0, "reverse": False},
    "fp": {"alpha": 10.0, "min_value": 0.25, "max_value": 1.00, "reverse": True},
    "fn": {"alpha": 10.0, "min_value": 1.00, "max_value": 1.00, "reverse": False},
}
# Default parameters for SentCLEME_independent
DEFAULT_CONFIG_SENTENCE_INDEPENDENT = {
    "tp": {"alpha": 10.0, "min_value": 2.50, "max_value": 10.0, "reverse": False},
    "fp": {"alpha": 10.0, "min_value": 0.25, "max_value": 1.00, "reverse": True},
    "fn": {"alpha": 10.0, "min_value": 1.00, "max_value": 1.00, "reverse": False},
}
