import copy
from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Dict, List, Tuple, Optional

from .utils import get_logger

LOGGER = get_logger(__name__)


@dataclass
class Edit(object):
    tgt_idx: int = field(
        default=0, metadata={"help": "Target index"}
    )
    src_interval: Optional[List[int]] = field(
        default=None, metadata={"help": "Source interval"}
    )
    tgt_interval: Optional[List[int]] = field(
        default=None, metadata={"help": "Target interval"}
    )
    src_tokens: Optional[List[str]] = field(
        default=None, metadata={"help": "Source tokens"}
    )
    tgt_tokens: Optional[List[str]] = field(
        default=None, metadata={"help": "Target tokens"}
    )
    src_tokens_tok: Optional[Any] = field(
        default=None, metadata={"help": "Source tokens tokenized by third toolkit"}
    )
    tgt_tokens_tok: Optional[Any] = field(
        default=None, metadata={"help": "Target tokens tokenized by third toolkit"}
    )
    type: Optional[List[str]] = field(
        default=None, metadata={"help": "Edit type"}
    )


@dataclass
class Chunk(Edit):
    chunk_idx: int = field(
        default=None, metadata={"help": "Chunk index"}
    )

    def __repr__(self):
        src_tokens = " ".join(self.src_tokens)
        tgt_tokens = " ".join(self.tgt_tokens)
        return f"Chunk(chunk_idx={self.chunk_idx}, type={self.type}, tgt_idx={self.tgt_idx}, " \
               f"{self.src_interval}: {src_tokens} -> {self.tgt_interval}: {tgt_tokens})"

    def __eq__(self, other):
        if self.chunk_idx == other.chunk_idx:
            if self.src_interval != other.src_interval or self.src_tokens != other.src_tokens:
                raise ValueError(f"Invalid edit comparison: {self} || {other}")
            elif self.tgt_tokens == other.tgt_tokens:
                return True
        return False

    def __hash__(self):
        return super().__hash__() + hash(self.chunk_idx)

    def is_insert_chunk(self) -> bool:
        return self.src_interval[0] == self.src_interval[1]


def chunk_list_to_text(chunks: List[Chunk], limiter=" "):
    src = limiter.join([' '.join(x.src_tokens) for x in chunks])
    tgt = limiter.join([' '.join(x.tgt_tokens) for x in chunks])
    return src, tgt


def all_correct(chunk_list: List[Chunk]):
    """ All chunks change the original text """
    for chunk in chunk_list:
        if not chunk.type:
            return False
    return True


def any_correct(chunk_list: List[Chunk]):
    """ Any chunks change the original text """
    for chunk in chunk_list:
        if chunk.type:
            return True
    return False


def map_parallel(
        src_tokens: List[str],
        edit_list: List[Edit],
) -> Dict[int, int]:
    """ Map source indices to target indices """
    token_mapping = {}
    src_idx, tgt_idx = 0, 0

    for edit in edit_list:
        token_mapping[edit.src_interval[0]] = edit.tgt_interval[0]
        token_mapping[edit.src_interval[1]] = edit.tgt_interval[1]
        while src_idx < edit.src_interval[0]:
            token_mapping[src_idx] = tgt_idx
            src_idx += 1
            tgt_idx += 1
        src_idx = edit.src_interval[1]
        tgt_idx = edit.tgt_interval[1]

    while src_idx <= len(src_tokens):
        token_mapping[src_idx] = tgt_idx
        src_idx += 1
        tgt_idx += 1
    return token_mapping


def merge_edit(
        src_tokens: List[str],
        tgt_tokens_list: List[List[str]],
        edits_list: List[List[Edit]],
        token_mapping_total: List[Dict[int, int]],
        merge_distance: int = 0,
) -> Tuple[List[List[Edit]], List[List[int]]]:
    """ Merge edits with overlapping or adjacent intervals into a single chunk
        :param src_tokens: Segmented source sentence
        :param tgt_tokens_list: Segmented target sentences
        :param edits_list: Edits given by target sentences
        :param token_mapping_total: Token mapping given by target sentences
        :param merge_distance: Distance of merging edits
        :return Tuple(
            merge_edits_list: Merged edits for targets,
            shared_interval_list: Shared edit intervals for targets,
        )
    """
    edits_flat = list(chain(*edits_list))
    sorted_edit_list = sorted(edits_flat, key=lambda x: x.src_interval[0])

    # Merge interval
    shared_interval_list = []
    for edit in sorted_edit_list:
        if shared_interval_list and edit.src_interval[0] <= shared_interval_list[-1][1] + merge_distance:
            # overlap
            shared_interval_list[-1][1] = max(edit.src_interval[1], shared_interval_list[-1][1])
        else:
            # non-overlap
            shared_interval_list.append(list(edit.src_interval))
    LOGGER.debug(f"shared_interval_list: {shared_interval_list}")

    def get_shared_interval(interval):
        for i in shared_interval_list:
            if interval[0] >= i[0] and interval[1] <= i[1]:
                return i
        return None

    # Construct token_mapping: src_token -> tgt_token
    merge_edits_list = []
    for tgt_idx in range(len(tgt_tokens_list)):
        token_mapping = token_mapping_total[tgt_idx]
        LOGGER.debug(f"{' '.join(src_tokens)} || {' '.join(tgt_tokens_list[tgt_idx])}")
        LOGGER.debug(f"token_mapping[{tgt_idx}]={token_mapping}")

        for edit in edits_list[tgt_idx]:
            ori_edit = copy.deepcopy(edit)
            shared_interval = get_shared_interval(edit.src_interval)
            assert shared_interval is not None

            if shared_interval[0] == 0:
                tgt_beg_idx = 0
            else:
                tgt_beg_idx = token_mapping[shared_interval[0] - 1] + 1
            tgt_end_idx = token_mapping[shared_interval[1]]

            edit.src_interval = shared_interval
            edit.tgt_interval = [tgt_beg_idx, tgt_end_idx]
            edit.src_tokens = src_tokens[shared_interval[0]: shared_interval[1]]
            edit.tgt_tokens = tgt_tokens_list[tgt_idx][tgt_beg_idx: tgt_end_idx]
            LOGGER.debug(f"Merge edits: {ori_edit} -> {edit}")

        # Deduplicate edits
        merge_edits = []
        for edit in edits_list[tgt_idx]:
            if edit not in merge_edits:
                merge_edits.append(edit)
            else:
                merge_edits[-1].type.extend(edit.type)
                LOGGER.debug(f"Merge the same edits: {merge_edits[-1]}")
        merge_edits_list.append(merge_edits.copy())
        LOGGER.debug(f"merge_edits: {merge_edits}")
    return merge_edits_list, shared_interval_list


def convert_edit_into_chunk(
        src_tokens: List[str],
        tgt_tokens_list: List[List[str]],
        merge_edits_list: List[List[Edit]],
        shared_interval_list: List[List[int]],
        token_mapping_total: List[Dict[int, int]],
) -> List[List[Chunk]]:
    """ Convert edits to chunks
        @param src_tokens: Segmented source sentence
        @param tgt_tokens_list: Segmented target sentences
        @param merge_edits_list: Merged edits for targets
        @param shared_interval_list: Shared edit intervals for targets
        @param token_mapping_total: Token mapping given by target sentences
        @return: Chunk sequences
    """
    valid_interval_list = []
    src_idx, interval_idx = 0, 0

    # Acquire chunk intervals
    while src_idx <= len(src_tokens):
        if interval_idx == len(shared_interval_list):
            if src_idx != len(src_tokens):
                valid_interval_list.append([src_idx, len(src_tokens)])
            break

        curr_interval = shared_interval_list[interval_idx]
        if src_idx == curr_interval[0] == curr_interval[1]:
            # Dummy Chunk
            valid_interval_list.append(curr_interval.copy())
            interval_idx += 1
        elif curr_interval[0] <= src_idx < curr_interval[1]:
            # Corrected Chunk
            valid_interval_list.append(curr_interval.copy())
            src_idx = curr_interval[1]
            interval_idx += 1
        else:
            # Unchanged Chunk
            valid_interval_list.append([src_idx, curr_interval[0]])
            src_idx = curr_interval[0]
    LOGGER.debug(f"valid_interval_list: {valid_interval_list}")

    # Add chunk
    chunk_list_total = []
    for tgt_idx in range(len(merge_edits_list)):
        tgt_sent = tgt_tokens_list[tgt_idx]
        edit_list = merge_edits_list[tgt_idx]
        token_mapping = token_mapping_total[tgt_idx]
        chunk_list = []
        for chunk_idx, chunk_interval in enumerate(valid_interval_list):
            is_ungrammatical = False
            for edit in edit_list:
                if chunk_interval == edit.src_interval:
                    # Corrected/Dummy Chunk
                    chunk_list.append(Chunk(
                        chunk_idx=chunk_idx,
                        tgt_idx=tgt_idx,
                        src_interval=edit.src_interval,
                        tgt_interval=edit.tgt_interval,
                        src_tokens=edit.src_tokens,
                        tgt_tokens=edit.tgt_tokens,
                        type=edit.type,
                    ))
                    is_ungrammatical = True
                    break
            if not is_ungrammatical:
                # Unchanged Chunk
                tgt_beg_idx = token_mapping[chunk_interval[0]]
                tgt_end_idx = 0 if chunk_interval[1] == 0 else token_mapping[chunk_interval[1] - 1] + 1
                chunk_list.append(Chunk(
                    chunk_idx=chunk_idx,
                    tgt_idx=tgt_idx,
                    src_interval=chunk_interval,
                    tgt_interval=[tgt_beg_idx, tgt_end_idx],
                    src_tokens=src_tokens[chunk_interval[0]: chunk_interval[1]],
                    tgt_tokens=tgt_sent[tgt_beg_idx: tgt_end_idx],
                    type=[],
                ))
        chunk_list_total.append(chunk_list.copy())
        LOGGER.debug(f"chunk_list: {chunk_list}")

    # Sanity check: all chunk_list should have the same length.
    chunk_len = len(chunk_list_total[0])
    for chunk_list in chunk_list_total:
        assert len(chunk_list) == chunk_len, f"{len(chunk_list)} != {chunk_len}"
    return chunk_list_total
