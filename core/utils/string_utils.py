import re
from string import punctuation
from typing import List, Union

# Punctuation sets for different languages
PUNCTUATION_ZHO = "！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏."

PUNCTUATION_ENG = punctuation

PUNCTUATION_OTH = "•・·"

PUNCTUATION = PUNCTUATION_ZHO + PUNCTUATION_ENG + PUNCTUATION_OTH

# Chinese text converter (Traditional to Simplified)
SIMPLIFIER = None

# Chinese Unicode character ranges
ZHO_UNICODE_RANGES = [
    ("\u3400", "\u4db5"),  # CJK Unified Ideographs Extension A, release 3.0
    ("\u4e00", "\u9fa5"),  # CJK Unified Ideographs, release 1.1
    ("\u9fa6", "\u9fbb"),  # CJK Unified Ideographs, release 4.1
    ("\uf900", "\ufa2d"),  # CJK Compatibility Ideographs, release 1.1
    ("\ufa30", "\ufa6a"),  # CJK Compatibility Ideographs, release 3.2
    ("\ufa70", "\ufad9"),  # CJK Compatibility Ideographs, release 4.1
    ("\u20000", "\u2a6d6"),  # (UTF16) CJK Unified Ideographs Extension B, release 3.1
    ("\u2f800", "\u2fa1d"),  # (UTF16) CJK Compatibility Supplement, release 3.1
    ("\uff00", "\uffef"),  # Full width ASCII, full width of English punctuation,
    # half width Katakana, half wide half width kana, Korean alphabet
    ("\u2e80", "\u2eff"),  # CJK Radicals Supplement
    ("\u3000", "\u303f"),  # CJK punctuation mark
    ("\u31c0", "\u31ef"),  # CJK stroke
    ("\u2f00", "\u2fdf"),  # Kangxi Radicals
    ("\u2ff0", "\u2fff"),  # Chinese character structure
    ("\u3100", "\u312f"),  # Phonetic symbols
    ("\u31a0", "\u31bf"),  # Phonetic symbols (Taiwanese and Hakka expansion)
    ("\ufe10", "\ufe1f"),
    ("\ufe30", "\ufe4f"),
    ("\u2600", "\u26ff"),
    ("\u2700", "\u27bf"),
    ("\u3200", "\u32ff"),
    ("\u3300", "\u33ff"),
]


def remove_space(batch: Union[str, List[str]]) -> Union[str, List[str]]:
    """Remove all whitespace characters from text.

    This function handles both single strings and lists of strings,
    removing all spaces including special Unicode spaces.

    Args:
        batch: Input string or list of strings

    Returns:
        String or list of strings with all spaces removed
    """

    def _remove_space(text: str):
        text = text.strip().replace("\u3000", " ").replace("\xa0", " ")
        text = "".join(text.split())
        return text

    if isinstance(batch, str):
        return _remove_space(batch)
    else:
        return [_remove_space(x) for x in batch]


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to a maximum length, adding ellipsis if needed.

    Args:
        text: Input text
        max_length: Maximum allowed length

    Returns:
        Truncated text with ellipsis if truncation occurred
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def simplify_chinese(text: str) -> str:
    """Convert traditional Chinese characters to simplified Chinese.

    Args:
        text: Input text containing traditional Chinese

    Returns:
        Text with traditional characters converted to simplified
    """
    if SIMPLIFIER is None:
        from opencc import OpenCC

        SIMPLIFIER = OpenCC("t2s")
    return SIMPLIFIER.convert(text)


def is_punct(text: str) -> bool:
    """Check if all characters in the text are punctuation.

    Args:
        text: Input text to check

    Returns:
        True if all characters are punctuation, False otherwise
    """
    return all([x in PUNCTUATION for x in text])


def is_chinese_char(uchar: str) -> bool:
    """Check if a single character is a Chinese character.

    Args:
        uchar: A single character to check

    Returns:
        True if the character is Chinese, False otherwise

    Raises:
        ValueError: If input is not a single character
    """
    if len(uchar) != 1:
        raise ValueError(f"Input must be char: {uchar}")

    # Only for simplified char set
    # return "\u4e00" <= uchar <= "\u9fff"

    for start, end in ZHO_UNICODE_RANGES:
        if start <= uchar <= end:
            return True
    return False


def all_chinese_chars(text: str) -> bool:
    """Check if all characters in the text are Chinese characters.

    Note that spaces and non-Chinese characters (like English letters)
    will cause this function to return False.

    Args:
        text: Input text to check

    Returns:
        True if all characters are Chinese, False otherwise
    """
    return all([is_chinese_char(ch) for ch in text])


def split_sentence(
    line: str, lang: str = "all", limit: int = 510, enable_spacy: bool = False, enable_blingfire: bool = False
) -> List[str]:
    """Split text into sentences based on punctuation and language rules.

    Args:
        line: Input text to split
        lang: Language mode - "all" for multilingual, "zh" for Chinese, "en" for English
        limit: Maximum length of each sentence
        enable_spacy: Whether to use spaCy for sentence splitting
        enable_blingfire: Whether to use BlingFire for sentence splitting

    Returns:
        List of sentences

    Raises:
        ValueError: If both enable_spacy and enable_blingfire are True
        NotImplementedError: If spaCy splitting is requested (not implemented)
    """
    line = line.strip()

    if all([enable_spacy, enable_blingfire]):
        raise ValueError("enable_spacy and enable_blingfire and not be both true.")

    if enable_spacy:
        # TODO: Implement sentence splitting using spaCy
        raise NotImplementedError()
    elif enable_blingfire:
        try:
            # BlingFire is a fast Finite State machine and Regular expression library
            # https://github.com/microsoft/BlingFire
            from blingfire import text_to_sentences
        except Exception:
            print("pip install blingfire")
        return text_to_sentences(line).split("\n")

    results = []
    try:
        if lang == "zho":
            # Chinese single-character sentence delimiters
            line = re.sub("(?P<quotation_mark>([。？！](?![”’\"'])))", r"\g<quotation_mark>\n", line)
            # Special quotation marks
            line = re.sub("(?P<quotation_mark>([。？！])[”’\"'])", r"\g<quotation_mark>\n", line)
        elif lang == "eng":
            # English single-character sentence delimiters
            line = re.sub("(?P<quotation_mark>([.?!](?![”’\"'])))", r"\g<quotation_mark>\n", line)
            # Special quotation marks
            line = re.sub("(?P<quotation_mark>([?!.][\"']))", r"\g<quotation_mark>\n", line)
        else:
            # Multi-language single-character sentence delimiters
            line = re.sub("(?P<quotation_mark>([。？！….?!](?![”’\"'])))", r"\g<quotation_mark>\n", line)
            # Special quotation marks
            line = re.sub("(?P<quotation_mark>(([。？！.!?]|…{1,2})[”’\"']))", r"\g<quotation_mark>\n", line)

        # Process each sentence, splitting if longer than limit
        for sent in line.splitlines():
            sent = sent.strip()
            if sent:
                while len(sent) > limit:
                    results.append(sent[:limit])
                    sent = sent[limit:]
                results.append(sent)
    except RuntimeError:
        # Fall back to returning the whole line if splitting fails
        results.clear()
        results.append(line)
    return results


def subword_align(src_line: str, tgt_line: str) -> List[int]:
    """Align subwords to their corresponding words, creating an alignment index.

    This function maps each subword token in the target sequence to its
    corresponding word position in the source sequence.

    Args:
        src_line: Source sequence made up of words
        tgt_line: Target sequence made up of subword tokens (with @@ markers)

    Returns:
        List of indices mapping each target token to its source word position

    Example:
        src_line: "Humans have many basic needs"
        tgt_line: "Hum@@ ans have many basic needs"
        return: [0, 0, 1, 2, 3, 4]
    """
    # Handle special Unicode spaces
    src_line = src_line.replace("\u3000", "u3000").replace("\xa0", "xa0")
    tgt_line = tgt_line.replace("\u3000", "u3000").replace("\xa0", "xa0")
    src_tokens, tgt_tokens = src_line.rstrip().split(), tgt_line.rstrip().split()

    if len(src_tokens) == 0:
        assert len(tgt_tokens) == 0
        return []

    i, j = 0, 0
    aligned_results = []
    try:
        while j < len(tgt_tokens):
            # Process subword tokens (ending with @@)
            while tgt_tokens[j].endswith("@@"):
                # Handle special cases
                if src_tokens[i].endswith("@@") and tgt_tokens[j] == "@@":
                    break
                if src_tokens[i] == "@@@@@" and tgt_tokens[j] == "@@@@":
                    break
                # Map this subword to the current source word
                aligned_results.append(i)
                j += 1
            # Map the final subword of the current word
            aligned_results.append(i)
            i += 1
            j += 1
    except RuntimeError:
        print(src_line, tgt_line)
        print(src_tokens)
        print(tgt_tokens)

    # Validate alignment
    assert len(aligned_results) == len(tgt_tokens)
    assert int(aligned_results[-1]) == len(src_tokens) - 1, f"{src_line}, {tgt_line}"
    return aligned_results
