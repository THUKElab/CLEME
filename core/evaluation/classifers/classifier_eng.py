from pathlib import Path

import spacy.symbols as POS
from errant.en.lancaster import LancasterStemmer
from rapidfuzz.distance import Levenshtein

from core.data.objects import Edit

from .classifier_base import BaseClassifier


class ClassifierEng(BaseClassifier):
    """English-specific error classifier implementing the BaseClassifier interface.

    This classifier handles English grammatical errors and classifies them into
    specific error types based on linguistic analysis.
    """

    def signature(self) -> str:
        return "eng"

    def __call__(self, source, target, edit: Edit):
        """Classifies grammatical errors in English text.

        Analyzes the edit operation and assigns appropriate error types based on
        linguistic patterns and rules specific to English grammar.

        Args:
            source: Original text tokens with POS information.
            target: Corrected text tokens with POS information.
            edit: Edit object containing token spans to be classified.

        Returns:
            Edit: The same Edit object with updated error type information.
        """
        # Nothing to nothing is a detected but not corrected edit
        if not edit.src_tokens_tok and not edit.tgt_tokens_tok:
            edit.types = ["UNK"]
        # Missing
        elif not edit.src_tokens_tok and edit.tgt_tokens_tok:
            op = "M:"
            cat = get_one_sided_type(edit.tgt_tokens_tok)
            edit.types = [op + cat]
        # Unnecessary
        elif edit.src_tokens_tok and not edit.tgt_tokens_tok:
            op = "U:"
            cat = get_one_sided_type(edit.src_tokens_tok)
            edit.types = [op + cat]
        # Replacement and special cases
        else:
            # Same to same is a detected but not corrected edit
            if edit.src_tokens == edit.tgt_tokens:
                edit.types = ["UNK"]
            # Special: Ignore case change at the end of multi token edits
            # E.g. [Doctor -> The doctor], [, since -> . Since]
            # Classify the edit as if the last token wasn't there
            elif edit.src_tokens_tok[-1].lower == edit.tgt_tokens_tok[-1].lower and (
                len(edit.src_tokens_tok) > 1 or len(edit.tgt_tokens_tok) > 1
            ):
                # Store a copy of the full orig and cor toks
                all_src_tokens_tok = edit.src_tokens_tok[:]
                all_tgt_tokens_tok = edit.tgt_tokens_tok[:]
                # Truncate the instance toks for classification
                edit.src_tokens_tok = edit.src_tokens_tok[:-1]
                edit.tgt_tokens_tok = edit.tgt_tokens_tok[:-1]
                # Classify the truncated edit
                edit = self(source, target, edit)
                # Restore the full orig and cor toks
                edit.src_tokens_tok = all_src_tokens_tok
                edit.tgt_tokens_tok = all_tgt_tokens_tok
            # Replacement
            else:
                op = "R:"
                cat = get_two_sided_type(edit.src_tokens_tok, edit.tgt_tokens_tok)
                edit.types = [op + cat]
        return edit


def load_word_list(path):
    """Loads a Hunspell word list from a file.

    Args:
        path: File path to the word list.

    Returns:
        set: A set containing all the words from the file.
    """
    with open(path) as word_list:
        return set([word.strip() for word in word_list])


def load_pos_map(path):
    """Loads Universal Dependency POS Tags mapping from a file.

    https://universaldependencies.org/tagset-conversion/en-penn-uposf.html

    Creates a dictionary that maps Penn Treebank tags to Universal Dependencies
    POS tags with some modifications for readability and consistency.

    Args:
        path: Path to the POS mapping file.

    Returns:
        dict: Dictionary mapping Penn tags to UD tags.
    """
    map_dict = {}
    with open(path) as map_file:
        for line in map_file:
            line = line.strip().split("\t")
            # Change ADP to PREP for readability
            if line[1] == "ADP":
                map_dict[line[0]] = "PREP"
            # Change PROPN to NOUN; we don't need a prop noun tag
            elif line[1] == "PROPN":
                map_dict[line[0]] = "NOUN"
            # Change CCONJ to CONJ
            elif line[1] == "CCONJ":
                map_dict[line[0]] = "CONJ"
            # Otherwise
            else:
                map_dict[line[0]] = line[1].strip()
        # Add some spacy PTB tags not in the original mapping.
        map_dict['""'] = "PUNCT"
        map_dict["SP"] = "SPACE"
        map_dict["_SP"] = "SPACE"
        map_dict["BES"] = "VERB"
        map_dict["HVS"] = "VERB"
        map_dict["ADD"] = "X"
        map_dict["GW"] = "X"
        map_dict["NFP"] = "X"
        map_dict["XX"] = "X"
    return map_dict


# Classifier resources
base_dir = Path(__file__).resolve().parent
# Spacy
nlp = None
# Lancaster Stemmer
stemmer = LancasterStemmer()
# GB English word list (inc -ise and -ize)
spell = load_word_list(base_dir / "../langs/eng/en_GB-large.txt")
# Part of speech map file
pos_map = load_pos_map(base_dir / "../langs/eng/en-ptb_map")
# Open class coarse Spacy POS tags
open_pos1 = {POS.ADJ, POS.ADV, POS.NOUN, POS.VERB}
# Open class coarse Spacy POS tags (strings)
open_pos2 = {"ADJ", "ADV", "NOUN", "VERB"}
# Rare POS tags that make uninformative error categories
rare_pos = {"INTJ", "NUM", "SYM", "X"}
# Contractions
conts = {"'d", "'ll", "'m", "n't", "'re", "'s", "'ve"}
# Special auxiliaries in contractions.
aux_conts = {"ca": "can", "sha": "shall", "wo": "will"}
# Some dep labels that map to pos tags.
dep_map = {
    "acomp": "ADJ",
    "amod": "ADJ",
    "advmod": "ADV",
    "det": "DET",
    "prep": "PREP",
    "prt": "PART",
    "punct": "PUNCT",
}


def get_edit_info(toks):
    """Extracts POS and dependency information from tokens.

    Args:
        toks: List of spaCy tokens.

    Returns:
        tuple: A tuple containing lists of POS tags and dependency relations.
    """
    pos = []
    dep = []
    for tok in toks:
        pos.append(pos_map[tok.tag_])
        dep.append(tok.dep_)
    return pos, dep


def get_one_sided_type(toks):
    """Determines error type based on tokens from only one side of the edit.

    Used when one side of the edit is null, so we can only use tokens from
    the other side to classify the error.

    Args:
        toks: List of spaCy tokens from one side of the edit.

    Returns:
        str: Error type classification.
    """
    # Special cases
    if len(toks) == 1:
        # Possessive noun suffixes; e.g. ' -> 's
        if toks[0].tag_ == "POS":
            return "NOUN:POSS"
        # Contractions. Rule must come after possessive
        if toks[0].lower_ in conts:
            return "CONTR"
        # Infinitival "to" is treated as part of a verb form
        if toks[0].lower_ == "to" and toks[0].pos == POS.PART and toks[0].dep_ != "prep":
            return "VERB:FORM"
    # Extract pos tags and parse info from the toks
    pos_list, dep_list = get_edit_info(toks)
    # Auxiliary verbs
    if set(dep_list).issubset({"aux", "auxpass"}):
        return "VERB:TENSE"
    # POS-based tags. Ignores rare, uninformative categories
    if len(set(pos_list)) == 1 and pos_list[0] not in rare_pos:
        return pos_list[0]
    # More POS-based tags using special dependency labels
    if len(set(dep_list)) == 1 and dep_list[0] in dep_map.keys():
        return dep_map[dep_list[0]]
    # To-infinitives and phrasal verbs
    if set(pos_list) == {"PART", "VERB"}:
        return "VERB"
    # Tricky cases
    else:
        return "OTHER"


def get_two_sided_type(src_tokens_tok, tgt_tokens_tok):
    """Determines error type based on both original and corrected tokens.

    Analyzes both sides of an edit to determine the most specific error type
    based on linguistic properties and patterns.

    Args:
        src_tokens_tok: List of spaCy tokens from the source text.
        tgt_tokens_tok: List of spaCy tokens from the target text.

    Returns:
        str: Error type classification.
    """
    # Extract pos tags and parse info from the toks as lists
    o_pos, o_dep = get_edit_info(src_tokens_tok)
    c_pos, c_dep = get_edit_info(tgt_tokens_tok)

    # Orthography; i.e. whitespace and/or case errors.
    if only_orth_change(src_tokens_tok, tgt_tokens_tok):
        return "ORTH"
    # Word Order; only matches exact reordering.
    if exact_reordering(src_tokens_tok, tgt_tokens_tok):
        return "WO"

    # 1:1 replacements (very common)
    if len(src_tokens_tok) == len(tgt_tokens_tok) == 1:
        # 1. SPECIAL CASES
        # Possessive noun suffixes; e.g. ' -> 's
        if src_tokens_tok[0].tag_ == "POS" or tgt_tokens_tok[0].tag_ == "POS":
            return "NOUN:POSS"
        # Contraction. Rule must come after possessive.
        if (src_tokens_tok[0].lower_ in conts or tgt_tokens_tok[0].lower_ in conts) and o_pos == c_pos:
            return "CONTR"
        # Special auxiliaries in contractions (1); e.g. ca -> can, wo -> will
        # Rule was broken in V1. Turned off this fix for compatibility.
        if (
            src_tokens_tok[0].lower_ in aux_conts and tgt_tokens_tok[0].lower_ == aux_conts[src_tokens_tok[0].lower_]
        ) or (
            tgt_tokens_tok[0].lower_ in aux_conts and src_tokens_tok[0].lower_ == aux_conts[tgt_tokens_tok[0].lower_]
        ):
            return "CONTR"
        # Special auxiliaries in contractions (2); e.g. ca -> could, wo -> should
        if src_tokens_tok[0].lower_ in aux_conts or tgt_tokens_tok[0].lower_ in aux_conts:
            return "VERB:TENSE"
        # Special: "was" and "were" are the only past tense SVA
        if {src_tokens_tok[0].lower_, tgt_tokens_tok[0].lower_} == {"was", "were"}:
            return "VERB:SVA"

        # 2. SPELLING AND INFLECTION
        # Only check alphabetical strings on the original side
        # Spelling errors take precedence over POS errors; this rule is ordered
        if src_tokens_tok[0].text.isalpha():
            # Check a GB English dict for both orig and lower case.
            # E.g. "cat" is in the dict, but "Cat" is not.
            if src_tokens_tok[0].text not in spell and src_tokens_tok[0].lower_ not in spell:
                # Check if both sides have a common lemma
                if src_tokens_tok[0].lemma == tgt_tokens_tok[0].lemma:
                    # Inflection; often count vs mass nouns or e.g. got vs getted
                    if o_pos == c_pos and o_pos[0] in {"NOUN", "VERB"}:
                        return o_pos[0] + ":INFL"
                    # Unknown morphology; i.e. we cannot be more specific.
                    else:
                        return "MORPH"
                # Use string similarity to detect true spelling errors.
                else:
                    # Normalised Lev distance works better than Lev ratio
                    str_sim = Levenshtein.normalized_similarity(src_tokens_tok[0].lower_, tgt_tokens_tok[0].lower_)
                    # WARNING: THIS IS AN APPROXIMATION.
                    # Thresholds tuned manually on FCE_train + W&I_train
                    # str_sim > 0.55 is almost always a true spelling error
                    if str_sim > 0.55:
                        return "SPELL"
                    # Special scores for shorter sequences are usually SPELL
                    if str_sim == 0.5 or round(str_sim, 3) == 0.333:
                        # Short strings are more likely to be spell: eles -> else
                        if len(src_tokens_tok[0].text) <= 4 and len(tgt_tokens_tok[0].text) <= 4:
                            return "SPELL"
                    # The remainder are usually word choice: amounght -> number
                    # Classifying based on cor_pos alone is generally enough.
                    if c_pos[0] not in rare_pos:
                        return c_pos[0]
                    # Anything that remains is OTHER
                    else:
                        return "OTHER"

        # 3. MORPHOLOGY
        # Only ADJ, ADV, NOUN and VERB can have inflectional changes.
        if src_tokens_tok[0].lemma == tgt_tokens_tok[0].lemma and o_pos[0] in open_pos2 and c_pos[0] in open_pos2:
            # Same POS on both sides
            if o_pos == c_pos:
                # Adjective form; e.g. comparatives
                if o_pos[0] == "ADJ":
                    return "ADJ:FORM"
                # Noun number
                if o_pos[0] == "NOUN":
                    return "NOUN:NUM"
                # Verbs - various types
                if o_pos[0] == "VERB":
                    # NOTE: These rules are carefully ordered.
                    # Use the dep parse to find some form errors.
                    # Main verbs preceded by aux cannot be tense or SVA.
                    if preceded_by_aux(src_tokens_tok, tgt_tokens_tok):
                        return "VERB:FORM"
                    # Use fine PTB tags to find various errors.
                    # FORM errors normally involve VBG or VBN.
                    if src_tokens_tok[0].tag_ in {"VBG", "VBN"} or tgt_tokens_tok[0].tag_ in {"VBG", "VBN"}:
                        return "VERB:FORM"
                    # Of what's left, TENSE errors normally involved VBD.
                    if src_tokens_tok[0].tag_ == "VBD" or tgt_tokens_tok[0].tag_ == "VBD":
                        return "VERB:TENSE"
                    # Of what's left, SVA errors normally involve VBZ.
                    if src_tokens_tok[0].tag_ == "VBZ" or tgt_tokens_tok[0].tag_ == "VBZ":
                        return "VERB:SVA"
                    # Any remaining aux verbs are called TENSE.
                    if o_dep[0].startswith("aux") and c_dep[0].startswith("aux"):
                        return "VERB:TENSE"
            # Use dep labels to find some more ADJ:FORM
            if set(o_dep + c_dep).issubset({"acomp", "amod"}):
                return "ADJ:FORM"
            # Adj to plural noun is usually noun number; e.g. musical -> musicals.
            if o_pos[0] == "ADJ" and tgt_tokens_tok[0].tag_ == "NNS":
                return "NOUN:NUM"
            # For remaining verb errors (rare), rely on c_pos
            if tgt_tokens_tok[0].tag_ in {"VBG", "VBN"}:
                return "VERB:FORM"
            if tgt_tokens_tok[0].tag_ == "VBD":
                return "VERB:TENSE"
            if tgt_tokens_tok[0].tag_ == "VBZ":
                return "VERB:SVA"
            # Tricky cases that all have the same lemma.
            else:
                return "MORPH"

        # Derivational morphology.
        if (
            stemmer.stem(src_tokens_tok[0].text) == stemmer.stem(tgt_tokens_tok[0].text)
            and o_pos[0] in open_pos2
            and c_pos[0] in open_pos2
        ):
            return "MORPH"

        # 4. GENERAL
        # Auxiliaries with different lemmas
        if o_dep[0].startswith("aux") and c_dep[0].startswith("aux"):
            return "VERB:TENSE"
        # POS-based tags. Some of these are context-sensitive misspellings.
        if o_pos == c_pos and o_pos[0] not in rare_pos:
            return o_pos[0]
        # Some dep labels map to POS-based tags.
        if o_dep == c_dep and o_dep[0] in dep_map.keys():
            return dep_map[o_dep[0]]
        # Phrasal verb particles.
        if set(o_pos + c_pos) == {"PART", "PREP"} or set(o_dep + c_dep) == {"prt", "prep"}:
            return "PART"
        # Can use dep labels to resolve DET + PRON combinations.
        if set(o_pos + c_pos) == {"DET", "PRON"}:
            # DET cannot be a subject or object.
            if c_dep[0] in {"nsubj", "nsubjpass", "dobj", "pobj"}:
                return "PRON"
            # "poss" indicates possessive determiner
            if c_dep[0] == "poss":
                return "DET"
        # NUM and DET are usually DET; e.g. a <-> one
        if set(o_pos + c_pos) == {"NUM", "DET"}:
            return "DET"
        # Special: other <-> another
        if {src_tokens_tok[0].lower_, tgt_tokens_tok[0].lower_} == {"other", "another"}:
            return "DET"
        # Special: your (sincerely) -> yours (sincerely)
        if src_tokens_tok[0].lower_ == "your" and tgt_tokens_tok[0].lower_ == "yours":
            return "PRON"
        # Special: no <-> not; this is very context sensitive
        if {src_tokens_tok[0].lower_, tgt_tokens_tok[0].lower_} == {"no", "not"}:
            return "OTHER"

        # 5. STRING SIMILARITY
        # These rules are quite language specific.
        if src_tokens_tok[0].text.isalpha() and tgt_tokens_tok[0].text.isalpha():
            # Normalised Lev distance works better than Lev ratio
            str_sim = Levenshtein.normalized_similarity(src_tokens_tok[0].lower_, tgt_tokens_tok[0].lower_)
            # WARNING: THIS IS AN APPROXIMATION.
            # Thresholds tuned manually on FCE_train + W&I_train
            # A. Short sequences are likely to be SPELL or function word errors
            if len(src_tokens_tok[0].text) == 1:
                # i -> in, a -> at
                if len(tgt_tokens_tok[0].text) == 2 and str_sim == 0.5:
                    return "SPELL"
            if len(src_tokens_tok[0].text) == 2:
                # in -> is, he -> the, to -> too
                if 2 <= len(tgt_tokens_tok[0].text) <= 3 and str_sim >= 0.5:
                    return "SPELL"
            if len(src_tokens_tok[0].text) == 3:
                # Special: the -> that (relative pronoun)
                if src_tokens_tok[0].lower_ == "the" and tgt_tokens_tok[0].lower_ == "that":
                    return "PRON"
                # Special: all -> everything
                if src_tokens_tok[0].lower_ == "all" and tgt_tokens_tok[0].lower_ == "everything":
                    return "PRON"
                # off -> of, too -> to, out -> our, now -> know
                if 2 <= len(tgt_tokens_tok[0].text) <= 4 and str_sim >= 0.5:
                    return "SPELL"
            # B. Longer sequences are also likely to include content word errors
            if len(src_tokens_tok[0].text) == 4:
                # Special: that <-> what
                if {src_tokens_tok[0].lower_, tgt_tokens_tok[0].lower_} == {"that", "what"}:
                    return "PRON"
                # Special: well <-> good
                if {src_tokens_tok[0].lower_, tgt_tokens_tok[0].lower_} == {"good", "well"} and c_pos[
                    0
                ] not in rare_pos:
                    return c_pos[0]
                # knew -> new,
                if len(tgt_tokens_tok[0].text) == 3 and str_sim > 0.5:
                    return "SPELL"
                # then <-> than, form -> from
                if len(tgt_tokens_tok[0].text) == 4 and str_sim >= 0.5:
                    return "SPELL"
                # gong -> going, hole -> whole
                if len(tgt_tokens_tok[0].text) == 5 and str_sim == 0.8:
                    return "SPELL"
                # high -> height, west -> western
                if len(tgt_tokens_tok[0].text) > 5 and str_sim > 0.5 and c_pos[0] not in rare_pos:
                    return c_pos[0]
            if len(src_tokens_tok[0].text) == 5:
                # Special: after -> later
                if {src_tokens_tok[0].lower_, tgt_tokens_tok[0].lower_} == {"after", "later"} and c_pos[
                    0
                ] not in rare_pos:
                    return c_pos[0]
                # where -> were, found -> fund
                if len(tgt_tokens_tok[0].text) == 4 and str_sim == 0.8:
                    return "SPELL"
                # thing <-> think, quite -> quiet, their <-> there
                if len(tgt_tokens_tok[0].text) == 5 and str_sim >= 0.6:
                    return "SPELL"
                # house -> domestic, human -> people
                if len(tgt_tokens_tok[0].text) > 5 and c_pos[0] not in rare_pos:
                    return c_pos[0]
            # C. Longest sequences include MORPH errors
            if len(src_tokens_tok[0].text) > 5 and len(tgt_tokens_tok[0].text) > 5:
                # Special: therefor -> therefore
                if src_tokens_tok[0].lower_ == "therefor" and tgt_tokens_tok[0].lower_ == "therefore":
                    return "SPELL"
                # Special: though <-> thought
                if {src_tokens_tok[0].lower_, tgt_tokens_tok[0].lower_} == {"though", "thought"}:
                    return "SPELL"
                # Morphology errors: stress -> stressed, health -> healthy
                if (
                    src_tokens_tok[0].text.startswith(tgt_tokens_tok[0].text)
                    or tgt_tokens_tok[0].text.startswith(src_tokens_tok[0].text)
                ) and str_sim >= 0.66:
                    return "MORPH"
                # Spelling errors: exiting -> exciting, wether -> whether
                if str_sim > 0.8:
                    return "SPELL"
                # Content word errors: learning -> studying, transport -> travel
                if str_sim < 0.55 and c_pos[0] not in rare_pos:
                    return c_pos[0]
                # NOTE: Errors between 0.55 and 0.8 are a mix of SPELL, MORPH and POS
        # Tricky cases
        else:
            return "OTHER"

    # Multi-token replacements (uncommon)
    # All auxiliaries
    if set(o_dep + c_dep).issubset({"aux", "auxpass"}):
        return "VERB:TENSE"
    # All same POS
    if len(set(o_pos + c_pos)) == 1:
        # Final verbs with the same lemma are tense; e.g. eat -> has eaten
        if o_pos[0] == "VERB" and src_tokens_tok[-1].lemma == tgt_tokens_tok[-1].lemma:
            return "VERB:TENSE"
        # POS-based tags.
        elif o_pos[0] not in rare_pos:
            return o_pos[0]
    # All same special dep labels.
    if len(set(o_dep + c_dep)) == 1 and o_dep[0] in dep_map.keys():
        return dep_map[o_dep[0]]
    # Infinitives, gerunds, phrasal verbs.
    if set(o_pos + c_pos) == {"PART", "VERB"}:
        # Final verbs with the same lemma are form; e.g. to eat -> eating
        if src_tokens_tok[-1].lemma == tgt_tokens_tok[-1].lemma:
            return "VERB:FORM"
        # Remaining edits are often verb; e.g. to eat -> consuming, look at -> see
        else:
            return "VERB"
    # Possessive nouns; e.g. friends -> friend 's
    if (o_pos == ["NOUN", "PART"] or c_pos == ["NOUN", "PART"]) and src_tokens_tok[0].lemma == tgt_tokens_tok[0].lemma:
        return "NOUN:POSS"
    # Adjective forms with "most" and "more"; e.g. more free -> freer
    if (
        (src_tokens_tok[0].lower_ in {"most", "more"} or tgt_tokens_tok[0].lower_ in {"most", "more"})
        and src_tokens_tok[-1].lemma == tgt_tokens_tok[-1].lemma
        and len(src_tokens_tok) <= 2
        and len(tgt_tokens_tok) <= 2
    ):
        return "ADJ:FORM"

    # Tricky cases.
    else:
        return "OTHER"


def only_orth_change(src_tokens_tok, tgt_tokens_tok):
    """Determines if the only differences between source and target tokens are whitespace or letter case.

    Joins all tokens after converting to lowercase and compares the resulting strings.

    Args:
        src_tokens_tok: Source tokens from Spacy
        tgt_tokens_tok: Target tokens from Spacy

    Returns:
        True if the differences are only orthographic (whitespace/case), False otherwise
    """
    o_join = "".join([o.lower_ for o in src_tokens_tok])
    c_join = "".join([c.lower_ for c in tgt_tokens_tok])
    if o_join == c_join:
        return True
    return False


def exact_reordering(src_tokens_tok, tgt_tokens_tok):
    """Checks if source and target tokens contain exactly the same elements but in different order.

    Sorts the lowercase tokens to compare their content regardless of order.

    Args:
        src_tokens_tok: Source tokens from Spacy
        tgt_tokens_tok: Target tokens from Spacy

    Returns:
        True if tokens are identical but reordered, False otherwise
    """
    # Sorting lets us keep duplicates.
    o_set = sorted([o.lower_ for o in src_tokens_tok])
    c_set = sorted([c.lower_ for c in tgt_tokens_tok])
    if o_set == c_set:
        return True
    return False


def preceded_by_aux(o_tok, c_tok):
    """Determines if both original and corrected tokens are preceded by auxiliary verbs.

    Checks dependency relationships to identify auxiliary verbs associated with the tokens.

    Args:
        o_tok: Original token from Spacy
        c_tok: Corrected token from Spacy

    Returns:
        True if both tokens have auxiliary verb dependencies, False otherwise
    """
    # If the tokens are aux, we need to check if they are the first aux.
    if o_tok[0].dep_.startswith("aux") and c_tok[0].dep_.startswith("aux"):
        # Find the parent verb
        o_head = o_tok[0].head
        c_head = c_tok[0].head
        # Find the children of the parent
        o_children = o_head.children
        c_children = c_head.children
        # Check the orig children.
        for o_child in o_children:
            # Look at the first aux...
            if o_child.dep_.startswith("aux"):
                # Check if the string matches o_tok
                if o_child.text != o_tok[0].text:
                    # If it doesn't, o_tok is not first so check cor
                    for c_child in c_children:
                        # Find the first aux in cor...
                        if c_child.dep_.startswith("aux"):
                            # If that doesn't match either, neither are first aux
                            if c_child.text != c_tok[0].text:
                                return True
                            # Break after the first cor aux
                            break
                # Break after the first orig aux.
                break
    # Otherwise, the toks are main verbs so we need to look for any aux.
    else:
        o_deps = [o_dep.dep_ for o_dep in o_tok[0].children]
        c_deps = [c_dep.dep_ for c_dep in c_tok[0].children]
        if "aux" in o_deps or "auxpass" in o_deps:
            if "aux" in c_deps or "auxpass" in c_deps:
                return True
    return False
