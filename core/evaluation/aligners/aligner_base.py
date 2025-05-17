from abc import ABC, abstractmethod
from typing import Any, List, Sequence, Tuple

import numpy as np


class BaseAligner(ABC):
    """Aligner provides the functionality of alignment between sentences.

    Args:
        del_cost (float): Cost of deletion.
        ins_cost (float): Cost of insertion.
        standard (bool): Whether use standard Standard Levenshtein. Default to False.
        brute_force (bool): Whether brute-force search all possible alignments. Defaults to False.
          Setting to Ture may introduce heavy computation.
    """

    def __init__(
        self,
        del_cost: float = 1.0,
        ins_cost: float = 1.0,
        standard: bool = False,
        brute_force: bool = False,
        verbose: bool = False,
    ) -> None:
        self.standard = standard
        self.del_cost = del_cost
        self.ins_cost = ins_cost
        self.brute_force = brute_force
        self.verbose = verbose
        self.align_seqs = None

    def signature(self) -> str:
        """Return a signature for the tokenizer."""
        raise NotImplementedError()

    def __call__(self, src_tokens: Sequence, tgt_tokens: Sequence) -> List[Tuple]:
        """Align the source and target tokens.

        Args:
            source (Sequence[Any]): Source sentence.
            target (Sequence[Any]): Target sentence.

        Returns:
            List[Tuple]: Alignment sequence [(op, o_start, o_end, c_start, c_end), ...]
        """
        # Align orig and cor and get the cost and op matrices
        cost_matrix, oper_matrix = self.align(src_tokens, tgt_tokens)
        # Get the cheapest align sequence from the op matrix
        align_seq = self.get_cheapest_align_seq(oper_matrix)

        if self.verbose:
            print(f"Source: {src_tokens}")
            print(f"Target: {tgt_tokens}")
            print(f"Cost Matrix: {cost_matrix}")
            print(f"Oper Matrix: {oper_matrix}")
            print(f"Alignment: {align_seq}")
            for a in align_seq:
                print(a[0], src_tokens[a[1] : a[2]], tgt_tokens[a[3] : a[4]])
        return align_seq

    def align(self, src_tokens: Sequence, tgt_tokens: Sequence) -> Tuple:
        """Aligns two sequences of tokens using a modified Levenshtein distance algorithm.

        This function computes the optimal alignment between source and target tokens by
        constructing a dynamic programming matrix that tracks the minimum edit distance and
        corresponding operations. The algorithm supports standard Levenshtein operations
        (insertion, deletion, substitution) as well as transpositions when enabled.

        The function handles both English and Chinese text through different token processing
        based on the language signature.

        Basic operations:
        - M: Match, which means the current character remains unchanged
        - D: Delete, which means the current character needs to be deleted
        - I: Insert, which means a character needs to be inserted at the current position
        - T: Transposition, which means a position operation involving character order issues

        Parameters:
        -----------
        src_tokens (Sequence): The source sequence of tokens to be aligned.
        tgt_tokens (Sequence): The target sequence of tokens to be aligned.

        Returns:
        --------
        Tuple: A tuple containing:
        - cost_matrix: A numpy array with the minimum edit costs for aligning subsequences.
        - oper_matrix: A numpy array with the operations needed for the optimal alignment.

        Notes:
        ------
        - The function uses different token comparison methods based on language (English or Chinese)
        - When self.brute_force is True, the operation matrix stores all operations with the same
        minimum cost at each position.
        - Transposition detection looks for permutations of characters within a window.
        - The standard Levenshtein mode uses fixed costs, while custom mode allows variable
        substitution costs through self.get_sub_cost().
        """
        # Create the cost matrix and the operation matrix
        cost_matrix = np.zeros((len(src_tokens) + 1, len(tgt_tokens) + 1))
        oper_matrix = np.full((len(src_tokens) + 1, len(tgt_tokens) + 1), "O", dtype=object)

        # Fill in the edges
        for i in range(1, len(src_tokens) + 1):
            cost_matrix[i][0] = cost_matrix[i - 1][0] + 1
            oper_matrix[i][0] = ["D"] if self.brute_force else "D"
        for j in range(1, len(tgt_tokens) + 1):
            cost_matrix[0][j] = cost_matrix[0][j - 1] + 1
            oper_matrix[0][j] = ["I"] if self.brute_force else "I"

        # Loop through the cost matrix
        for i in range(len(src_tokens)):
            for j in range(len(tgt_tokens)):
                if self.signature() == "eng":
                    src_token, tgt_token = src_tokens[i].orth, tgt_tokens[j].orth
                elif self.signature() == "zho":
                    src_token, tgt_token = src_tokens[i][0], tgt_tokens[j][0]
                else:
                    raise NotImplementedError(f"signature: {self.signature()}")

                if src_token == tgt_token:  # Match
                    cost_matrix[i + 1][j + 1] = cost_matrix[i][j]
                    oper_matrix[i + 1][j + 1] = ["M"] if self.brute_force else "M"
                else:  # Non-match
                    del_cost = cost_matrix[i][j + 1] + self.del_cost
                    ins_cost = cost_matrix[i + 1][j] + self.ins_cost
                    trans_cost = float("inf")
                    if self.standard:  # Standard Levenshtein (S = 1)
                        sub_cost = cost_matrix[i][j] + 1
                    else:
                        # Custom substitution
                        sub_cost = cost_matrix[i][j] + self.get_sub_cost(src_tokens[i], tgt_tokens[j])
                        # Transpositions require >=2 tokens
                        # Traverse the diagonal while there is not a Match.
                        k = 1
                        while (
                            i - k >= 0 and j - k >= 0 and cost_matrix[i - k + 1][j - k + 1] != cost_matrix[i - k][j - k]
                        ):
                            if self.signature() == "eng":
                                p1 = sorted([a.lower for a in src_tokens[i - k : i + 1]])
                                p2 = sorted([b.lower for b in tgt_tokens[j - k : j + 1]])
                            elif self.signature() == "zho":
                                p1 = sorted([a[0] for a in src_tokens[i - k : i + 1]])
                                p2 = sorted([b[0] for b in tgt_tokens[j - k : j + 1]])
                            else:
                                raise NotImplementedError(f"signature: {self.signature()}")
                            if p1 == p2:
                                trans_cost = cost_matrix[i - k][j - k] + k
                                break
                            k += 1
                    costs = [trans_cost, sub_cost, ins_cost, del_cost]
                    # Get the index of the cheapest (first cheapest if tied)
                    min_index = costs.index(min(costs))
                    # Save the cost and the op in the matrices
                    cost_matrix[i + 1][j + 1] = costs[min_index]

                    if not self.brute_force:
                        if min_index == 0:
                            oper_matrix[i + 1][j + 1] = "T" + str(k + 1)
                        elif min_index == 1:
                            oper_matrix[i + 1][j + 1] = "S"
                        elif min_index == 2:
                            oper_matrix[i + 1][j + 1] = "I"
                        else:
                            oper_matrix[i + 1][j + 1] = "D"
                    else:
                        for idx, cost in enumerate(costs):
                            if cost == costs[min_index]:
                                if idx == 0:
                                    if oper_matrix[i + 1][j + 1] == "O":
                                        oper_matrix[i + 1][j + 1] = ["T" + str(k + 1)]
                                    else:
                                        oper_matrix[i + 1][j + 1].append("T" + str(k + 1))
                                elif idx == 1:
                                    if oper_matrix[i + 1][j + 1] == "O":
                                        oper_matrix[i + 1][j + 1] = ["S"]
                                    else:
                                        oper_matrix[i + 1][j + 1].append("S")
                                elif idx == 2:
                                    if oper_matrix[i + 1][j + 1] == "O":
                                        oper_matrix[i + 1][j + 1] = ["I"]
                                    else:
                                        oper_matrix[i + 1][j + 1].append("I")
                                else:
                                    if oper_matrix[i + 1][j + 1] == "O":
                                        oper_matrix[i + 1][j + 1] = ["D"]
                                    else:
                                        oper_matrix[i + 1][j + 1].append("D")
        return cost_matrix, oper_matrix

    def get_cheapest_align_seq(self, oper_matrix: np.ndarray) -> List[Tuple]:
        """Retrieve the editing sequence with the smallest cost through backtracking.

        Args:
            oper_matrix (np.ndarray): 2-dimension operation matrix.

        Returns:
            List[Tuple]: [(op, o_start, o_end, c_start, c_end), ...]
        """
        if self.brute_force:  # BF search
            return self.get_cheapest_align_seq_bf(oper_matrix)

        align_seq = []
        i = oper_matrix.shape[0] - 1
        j = oper_matrix.shape[1] - 1
        # Work backwards from bottom right until we hit top left
        while i + j != 0:
            # Get the edit operation in the current cell
            op = oper_matrix[i][j]
            if op in {"M", "S"}:  # Matches and substitutions
                align_seq.append((op, i - 1, i, j - 1, j))
                i -= 1
                j -= 1
            elif op == "D":  # Deletions
                align_seq.append((op, i - 1, i, j, j))
                i -= 1
            elif op == "I":  # Insertions
                align_seq.append((op, i, i, j - 1, j))
                j -= 1
            else:  # Transpositions
                # Get the size of the transposition
                k = int(op[1:])
                align_seq.append((op, i - k, i, j - k, j))
                i -= k
                j -= k
        # Reverse the list to go from left to right and return
        align_seq.reverse()
        return align_seq

    def get_cheapest_align_seq_bf(self, oper_matrix: np.ndarray) -> List[Tuple]:
        self.align_seqs = []
        i = oper_matrix.shape[0] - 1
        j = oper_matrix.shape[1] - 1
        if abs(i - j) > 10:
            self._dfs(i, j, [], oper_matrix, strategy="first")
        else:
            self._dfs(i, j, [], oper_matrix, strategy="all")
        final_align_seqs = [seq[::-1] for seq in self.align_seqs]
        return final_align_seqs

    def _dfs(self, i: int, j: int, align_seq: List[Tuple], oper_matrix: np.ndarray, strategy: str = "all") -> None:
        """Apply depth-first search to find alignment sequences with minimum edit cost.

        This function recursively explores the operation matrix to build alignment sequences
        between source and target texts. When both indices reach zero, a complete alignment
        is found and stored. The function processes different operations (Match, Delete, Insert,
        Transposition) by adjusting indices and building the alignment sequence.

        Args:
            i (int): Current index in the source tokens.
            j (int): Current index in the target tokens.
            align_seq (List[Tuple]): Current alignment sequence being built.
            oper_matrix (np.ndarray): Matrix containing available operations at each position.
            strategy (str, optional): DFS Search strategy. Defaults to "all".
              - `all`: return all sequences with the same minimum cost.
              - `first`: return the firstly searched sequence with the minimum cost.

        Returns:
            None: Results are stored in self.align_seqs.
        """
        if i + j == 0:
            self.align_seqs.append(align_seq)
            return

        if strategy == "all":
            ops = oper_matrix[i][j]
        else:
            ops = oper_matrix[i][j][:-1]

        for op in ops:
            if op in {"M", "S"}:
                self._dfs(i - 1, j - 1, align_seq + [(op, i - 1, i, j - 1, j)], oper_matrix, strategy)
            elif op == "D":
                self._dfs(i - 1, j, align_seq + [(op, i - 1, i, j, j)], oper_matrix, strategy)
            elif op == "I":
                self._dfs(i, j - 1, align_seq + [(op, i, i, j - 1, j)], oper_matrix, strategy)
            else:
                k = int(op[1:])
                self._dfs(i - k, j - k, align_seq + [(op, i - k, i, j - k, j)], oper_matrix, strategy)

    @abstractmethod
    def get_sub_cost(self, src_token: Any, tgt_token: Any) -> float:
        raise NotImplementedError()
