from string import punctuation

PUNCTUATION_ZHO_END = "。？！～”…"
PUNCTUATION_ZHO_END_DOT = "。？！"
PUNCTUATION_ZHO_INNER_DOT = "，、；："
PUNCTUATION_ZHO_DOT = PUNCTUATION_ZHO_END_DOT + PUNCTUATION_ZHO_INNER_DOT

PUNCTUATION_ZHO_QUOTE = "‘’“”"
PUNCTUATION_ZHO_BRACKET = "（）《》〈〉"
PUNCTUATION_ZHO_CONNECTION = "—～"
PUNCTUATION_ZHO = "！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏."

PUNCTUATION_ENG = punctuation
PUNCTUATION_OTH = "•・·"
PUNCTUATION = PUNCTUATION_ZHO + PUNCTUATION_ENG + PUNCTUATION_OTH


# EditMetric
KEY_TP = "tp"
KEY_TN = "tn"
KEY_FP = "fp"
KEY_FN = "fn"

KEY_TP_EDIT = "tp_edit"
KEY_TN_EDIT = "tn_edit"
KEY_FP_EDIT = "fp_edit"
KEY_FN_EDIT = "fn_edit"
DELIMITER_M2 = "|||"

KEY_P = "P"
KEY_R = "R"
KEY_F = "F"
KEY_ACC = "Acc"

# NGramMetric
KEY_HYP_LEN = "hyp_len"
KEY_REF_LEN = "ref_len"
KEY_NGRAMS = "ngrams"
