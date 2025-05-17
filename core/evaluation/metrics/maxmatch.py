import sys
from copy import deepcopy
from typing import Dict, List, Tuple

from core.data.objects import Sample
from core.utils import get_logger

from ..scorers import BaseScorer, ScorerType
from ..tokenizers import BaseTokenizer
from .base import BaseEditMetric

KEY_TP = "tp"  # True Positive
KEY_TN = "tn"  # True Negative
KEY_FP = "fp"  # False Positive
KEY_FN = "fn"  # False Negative

LOGGER = get_logger(__name__)


class MaxMatch(BaseEditMetric):
    """Implementation of the MaxMatch algorithm for grammatical error correction evaluation.

    MaxMatch uses dynamic programming to find the optimal alignment between source, candidate,
    and reference texts, and calculates precision, recall, and F-score based on edit operations.

    Args:
        lang (str): Language code for tokenization
        tokenizer (str, optional): Tokenizer name. Defaults to language code if None.
        scorer (str, optional): Scoring method. Defaults to "corpus".
        enable_tqdm (bool, optional): Whether to show progress bars. Defaults to False.
        **kwargs: Additional arguments including:
          - candidate_max_unchanged_words (int): Maximum unchanged words in candidate edits. Defaults to 2.
          - reference_max_unchanged_words (int): Maximum unchanged words in reference edits. Defaults to 0.
          - ignore_whitespace_casing (bool): Whether to ignore whitespace and case differences. Defaults to False.
    """

    def __init__(
        self,
        lang: str,
        scorer: BaseScorer = None,
        scorer_type: ScorerType = None,
        tokenizer: BaseTokenizer = None,
        enable_tqdm: bool = True,
        **kwargs,
    ):
        super(MaxMatch, self).__init__(
            scorer=scorer, scorer_type=scorer_type, tokenizer=tokenizer, enable_tqdm=enable_tqdm
        )
        self.candidate_max_unchanged_words = kwargs.get("candidate_max_unchanged_words", 2)
        self.reference_max_unchanged_words = kwargs.get("reference_max_unchanged_words", 0)
        self.ignore_whitespace_casing = kwargs.get("ignore_whitespace_casing", False)

    def evaluate_sample(self, sample_hyp: Sample, sample_ref: Sample, **kwargs) -> List[Dict]:
        """Evaluates a sample by comparing hypothesis edits with reference edits.

        Args:
            sample_hyp (Sample): Hypothesis sample containing source and target texts
            sample_ref (Sample): Reference sample containing source and target texts

        Returns:
            List[Dict]: Evaluation results with TP, FP, FN, and TN counts for each reference

        Raises:
            AssertionError: If source texts don't match or sample structure is invalid
        """
        assert len(sample_hyp.source) == len(sample_hyp.target) == len(sample_ref.source) == 1
        assert (
            sample_hyp.source[0] == sample_ref.source[0]
        ), f"Source Not Equal: {sample_hyp.source[0]} || {sample_ref.source[0]}"

        source = sample_hyp.source[0].strip()
        candidate = sample_hyp.target[0].strip()
        references = list(map(str.strip, sample_ref.target))

        # Tokenize source and candidate texts
        source_tok = self.tokenizer(source, plain=True)
        candidate_tok = self.tokenizer(candidate, plain=True)
        LOGGER.debug(f"Source: {source}\nCandidate: {candidate}\nReferences{references}")

        # Generate edit graph for candidate
        V, E, dist, edits = self.get_graph(source_tok, candidate_tok, self.candidate_max_unchanged_words)

        # Retrieve reference edits if available
        ref_edits = sample_ref.edits[0] if sample_ref.edits else None

        # Process each reference target
        result = []
        for index, reference in enumerate(references):
            # Get reference edit sequence
            if ref_edits:
                ref_edit_seq = []
                for edit in ref_edits[index]:
                    ref_edit_seq.append(
                        (
                            edit.src_interval[0],
                            edit.src_interval[1],
                            " ".join(edit.src_tokens),
                            [" ".join(edit.tgt_tokens)],
                        )
                    )
            else:
                # Generate edit sequence from source and reference
                reference_tok = self.tokenizer(reference, plain=True)
                ref_edit_seq = self.get_graph_edit_seq(source_tok, reference_tok, self.reference_max_unchanged_words)
                ref_edit_seq = [(*item[:-1], [item[-1]]) for item in reversed(ref_edit_seq)]

            # Get candidate edit sequence and match with reference
            candidate_edit_seq = self.get_edit_seq(V, E, dist, edits, ref_edit_seq)
            correct = self.matchSeq(candidate_edit_seq, ref_edit_seq)
            LOGGER.debug(
                f"{candidate_edit_seq=}, {len(candidate_edit_seq)=}\n"
                f"{ref_edit_seq=}, {len(ref_edit_seq)=}\n"
                f"{correct=}, {len(correct)=}"
            )

            # Calculate evaluation metrics
            result.append(
                {
                    KEY_TP: len(correct),  # True Positives: edits correctly identified
                    KEY_FP: len(candidate_edit_seq) - len(correct),  # False Positives: incorrect edits
                    KEY_FN: len(ref_edit_seq) - len(correct),  # False Negatives: missed edits
                    KEY_TN: 0,  # True Negatives: not applicable for this metric
                }
            )
        return result

    def get_graph_edit_seq(self, source_tok: List[str], target_tok: List[str], max_unchanged_words: int):
        """Gets edit sequence from source to target using graph-based approach.

        Args:
            source_tok (List[str]): Tokenized source text
            target_tok (List[str]): Tokenized target text
            max_unchanged_words (int): Maximum number of unchanged words allowed

        Returns:
            List: Edit sequence from source to target
        """
        return self.get_edit_seq(*self.get_graph(source_tok, target_tok, max_unchanged_words))

    def get_graph(self, source_tok: List[str], target_tok: List[str], max_unchanged_words: int):
        """Constructs edit graph between source and target texts.

        Creates two Levenshtein matrices with different costs, builds edit graphs,
        merges them, and adds transitive arcs.

        Args:
            source_tok (List[str]): Tokenized source text
            target_tok (List[str]): Tokenized target text
            max_unchanged_words (int): Maximum number of unchanged words allowed

        Returns:
            Tuple: (V, E, dist, edits) where:
                - V: List of vertices
                - E: List of edges
                - dist: Dictionary of distances
                - edits: Dictionary of edit operations
        """
        # Create two Levenshtein matrices with different substitution costs
        lmatrix1, backpointers1 = self.levenshtein_matrix(source_tok, target_tok, 1, 1, 1)
        lmatrix2, backpointers2 = self.levenshtein_matrix(source_tok, target_tok, 1, 1, 2)

        # Build edit graphs from matrices
        V1, E1, dist1, edits1 = self.edit_graph(lmatrix1, backpointers1)
        V2, E2, dist2, edits2 = self.edit_graph(lmatrix2, backpointers2)

        # Merge the two graphs
        V, E, dist, edits = self.merge_graph(V1, V2, E1, E2, dist1, dist2, edits1, edits2)

        # Add transitive arcs to the graph
        V, E, dist, edits = self.transitive_arcs(V, E, dist, edits, max_unchanged_words)
        return V, E, dist, edits

    def get_edit_seq(self, V, E, dist, edits, gold=[]):
        """Gets the optimal edit sequence from the edit graph.

        Args:
            V (List): List of vertices
            E (List): List of edges
            dist (Dict): Dictionary of distances
            edits (Dict): Dictionary of edit operations
            gold (List, optional): Gold standard edits for weight setting. Defaults to [].

        Returns:
            List: Optimal edit sequence
        """

        # Function to ignore whitespace and case when comparing strings
        equals_ignore_whitespace_casing = lambda a, b: a.replace(" ", "").lower() == b.replace(" ", "").lower()

        # Set weights based on gold standard
        dist = self.set_weights(E, dist, edits, gold)

        # Find best edit sequence using Bellman-Ford algorithm
        edit_seq = self.best_edit_seq_bf(V, E, dist, edits)

        # Filter out edits that only differ in whitespace or casing if specified
        if self.ignore_whitespace_casing:
            edit_seq = [x for x in edit_seq if not equals_ignore_whitespace_casing(x[2], x[3])]

        return edit_seq

    def levenshtein_matrix(
        self, first: List[str], second: List[str], cost_ins: float = 1, cost_del: float = 1, cost_sub: float = 2
    ):
        """Computes Levenshtein distance matrix and backpointers between two token sequences.

        Args:
            first (List[str]): First token sequence
            second (List[str]): Second token sequence
            cost_ins (int, optional): Cost of insertion. Defaults to 1.
            cost_del (int, optional): Cost of deletion. Defaults to 1.
            cost_sub (int, optional): Cost of substitution. Defaults to 2.

        Returns:
            Tuple: (distance_matrix, backpointers) where:
              - distance_matrix: 2D matrix of edit distances
              - backpointers: Dictionary mapping positions to previous positions and edits
        """
        first_length = len(first) + 1
        second_length = len(second) + 1

        # Initialize matrices
        distance_matrix = [[None] * second_length for x in range(first_length)]
        backpointers = {}
        distance_matrix[0][0] = 0

        # Fill first column (deletions from source)
        for i in range(1, first_length):
            distance_matrix[i][0] = i
            edit = ("del", i - 1, i, first[i - 1], "", 0)
            backpointers[(i, 0)] = [((i - 1, 0), edit)]

        # Fill first row (insertions into source)
        for j in range(1, second_length):
            distance_matrix[0][j] = j
            # Always insert from the beginning
            edit = ("ins", 0, 0, "", second[j - 1], 0)
            # edit = ("ins", j-1, j-1, '', second[j-1], 0)
            backpointers[(0, j)] = [((0, j - 1), edit)]

        # Fill the rest of the matrix
        for i in range(1, first_length):
            for j in range(1, second_length):
                deletion = distance_matrix[i - 1][j] + cost_del
                insertion = distance_matrix[i][j - 1] + cost_ins

                # Substitution cost depends on whether tokens match
                if first[i - 1] == second[j - 1]:
                    substitution = distance_matrix[i - 1][j - 1]
                else:
                    substitution = distance_matrix[i - 1][j - 1] + cost_sub

                # Handle substitution/match case
                if substitution == min(substitution, deletion, insertion):
                    distance_matrix[i][j] = substitution
                    if first[i - 1] != second[j - 1]:
                        edit = ("sub", i - 1, i, first[i - 1], second[j - 1], 0)
                    else:
                        edit = ("noop", i - 1, i, first[i - 1], second[j - 1], 1)
                    try:
                        backpointers[(i, j)].append(((i - 1, j - 1), edit))
                    except KeyError:
                        backpointers[(i, j)] = [((i - 1, j - 1), edit)]

                # Handle deletion case
                if deletion == min(substitution, deletion, insertion):
                    distance_matrix[i][j] = deletion
                    edit = ("del", i - 1, i, first[i - 1], "", 0)
                    try:
                        backpointers[(i, j)].append(((i - 1, j), edit))
                    except KeyError:
                        backpointers[(i, j)] = [((i - 1, j), edit)]

                # Handle insertion case
                if insertion == min(substitution, deletion, insertion):
                    distance_matrix[i][j] = insertion
                    edit = ("ins", i, i, "", second[j - 1], 0)
                    try:
                        backpointers[(i, j)].append(((i, j - 1), edit))
                    except KeyError:
                        backpointers[(i, j)] = [((i, j - 1), edit)]
        return (distance_matrix, backpointers)

    def edit_graph(self, levi_matrix: List[List[int]], backpointers: Dict):
        """Constructs an edit graph from a Levenshtein matrix and backpointers.

        Uses breadth-first search to traverse the backpointers and build a graph
        representing possible edit paths.

        Args:
            levi_matrix (List[List[int]]): Levenshtein distance matrix
            backpointers (Dict): Dictionary of backpointers from Levenshtein matrix

        Returns:
            Tuple: (V, E, dist, edits) where:
              - V: List of vertices (matrix positions)
              - E: List of edges (connections between positions)
              - dist: Dictionary mapping edges to distances
              - edits: Dictionary mapping edges to edit operations
        """

        V = []  # Vertices
        E = []  # Edges
        dist = {}  # Edge distances
        edits = {}  # Edge edit operations

        # Start breadth-first search from bottom-right corner of matrix
        v_start = (len(levi_matrix) - 1, len(levi_matrix[0]) - 1)
        queue = [v_start]
        while len(queue) > 0:
            v = queue[0]
            queue = queue[1:]

            # Skip if vertex already processed
            if v in V:
                continue
            V.append(v)

            # Process all backpointers from current vertex
            try:
                for vnext_edits in backpointers[v]:
                    # Next vertex
                    vnext = vnext_edits[0]
                    # Edit operation
                    edit_next = vnext_edits[1]

                    # Add edge from next vertex to current vertex
                    E.append((vnext, v))
                    dist[(vnext, v)] = 1
                    edits[(vnext, v)] = edit_next

                    # Add next vertex to queue if not already there
                    if vnext not in queue:
                        queue.append(vnext)
            except KeyError:
                pass
        return (V, E, dist, edits)

    def transitive_arcs(self, V: List, E: List, dist: Dict, edits: Dict, max_unchanged_words: int):
        """
        Adds transitive arcs to the edit graph to allow skipping multiple tokens.

        This creates shortcuts in the graph by connecting vertices that can be reached
        through multiple steps, as long as they don't exceed the maximum number of
        unchanged words.

        Args:
            V (List): List of vertices
            E (List): List of edges
            dist (Dict): Dictionary of distances
            edits (Dict): Dictionary of edit operations
            max_unchanged_words (int): Maximum number of unchanged words allowed in a transitive arc

        Returns:
            Tuple: (V, E, dist, edits) with added transitive arcs
        """

        def get_distance(dist: Dict, v1, v2):
            """Helper function to get distance between vertices."""
            return dist.get((v1, v2), float("inf"))

        # Floyd-Warshall algorithm to find transitive arcs
        for k in range(len(V)):
            vk = V[k]
            for i in range(len(V)):
                vi = V[i]
                try:
                    eik = edits[(vi, vk)]
                except KeyError:
                    continue
                for j in range(len(V)):
                    vj = V[j]
                    try:
                        ekj = edits[(vk, vj)]
                    except KeyError:
                        continue

                    # Check if path through k is shorter
                    dik = get_distance(dist, vi, vk)
                    dkj = get_distance(dist, vk, vj)

                    if dik + dkj < get_distance(dist, vi, vj):
                        # Merge edits and add transitive arc if it doesn't exceed max_unchanged_words
                        eij = self.merge_edits(eik, ekj)
                        if eij[-1] <= max_unchanged_words:
                            E.append((vi, vj))
                            dist[(vi, vj)] = dik + dkj
                            edits[(vi, vj)] = eij

        # Remove noop transitive arcs (no-change operations that span multiple tokens)
        for edge in E:
            e = edits[edge]
            if e[0] == "noop" and dist[edge] > 1:
                E.remove(edge)
                dist[edge] = float("inf")
                del edits[edge]
        return (V, E, dist, edits)

    def merge_edits(self, e1: Tuple, e2: Tuple, joiner: str = " "):
        """Merges two consecutive edit operations into a single operation.

        Handles all possible combinations of edit types (insertion, deletion,
        substitution, and no-operation) when merging.

        Args:
            e1 (Tuple): First edit operation
            e2 (Tuple): Second edit operation
            joiner (str, optional): String used to join text. Defaults to " ".

        Returns:
            Tuple: Merged edit operation

        Raises:
            ValueError: If edit types are invalid
        """

        e = None
        if e1[0] == "ins":
            # Handle insertion followed by another operation
            if e2[0] == "ins":
                e = ("ins", e1[1], e2[2], "", e1[4] + joiner + e2[4], e1[5] + e2[5])
            elif e2[0] == "del":
                e = ("sub", e1[1], e2[2], e2[3], e1[4], e1[5] + e2[5])
            elif e2[0] == "sub":
                e = ("sub", e1[1], e2[2], e2[3], e1[4] + joiner + e2[4], e1[5] + e2[5])
            elif e2[0] == "noop":
                e = ("sub", e1[1], e2[2], e2[3], e1[4] + joiner + e2[4], e1[5] + e2[5])
        elif e1[0] == "del":
            # Handle deletion followed by another operation
            if e2[0] == "ins":
                e = ("sub", e1[1], e2[2], e1[3], e2[4], e1[5] + e2[5])
            elif e2[0] == "del":
                e = ("del", e1[1], e2[2], e1[3] + joiner + e2[3], "", e1[5] + e2[5])
            elif e2[0] == "sub":
                e = ("sub", e1[1], e2[2], e1[3] + joiner + e2[3], e2[4], e1[5] + e2[5])
            elif e2[0] == "noop":
                e = ("sub", e1[1], e2[2], e1[3] + joiner + e2[3], e2[4], e1[5] + e2[5])
        elif e1[0] == "sub":
            # Handle substitution followed by another operation
            if e2[0] == "ins":
                e = ("sub", e1[1], e2[2], e1[3], e1[4] + joiner + e2[4], e1[5] + e2[5])
            elif e2[0] == "del":
                e = ("sub", e1[1], e2[2], e1[3] + joiner + e2[3], e1[4], e1[5] + e2[5])
            elif e2[0] == "sub":
                e = ("sub", e1[1], e2[2], e1[3] + joiner + e2[3], e1[4] + joiner + e2[4], e1[5] + e2[5])
            elif e2[0] == "noop":
                e = ("sub", e1[1], e2[2], e1[3] + joiner + e2[3], e1[4] + joiner + e2[4], e1[5] + e2[5])
        elif e1[0] == "noop":
            # Handle no-operation followed by another operation
            if e2[0] == "ins":
                e = ("sub", e1[1], e2[2], e1[3], e1[4] + joiner + e2[4], e1[5] + e2[5])
            elif e2[0] == "del":
                e = ("sub", e1[1], e2[2], e1[3] + joiner + e2[3], e1[4], e1[5] + e2[5])
            elif e2[0] == "sub":
                e = ("sub", e1[1], e2[2], e1[3] + joiner + e2[3], e1[4] + joiner + e2[4], e1[5] + e2[5])
            elif e2[0] == "noop":
                e = ("noop", e1[1], e2[2], e1[3] + joiner + e2[3], e1[4] + joiner + e2[4], e1[5] + e2[5])
        else:
            raise ValueError
        return e

    def set_weights(self, E, dist, edits, gold_edits):
        """Sets weights for edges in the edit graph based on gold standard edits.

        Edges that match gold edits get negative weights (preferred), while
        other edges get small positive weights (penalized).

        Args:
            E (List): List of edges
            dist (Dict): Dictionary of distances
            edits (Dict): Dictionary of edit operations
            gold_edits (List): Gold standard edits

        Returns:
            Dict: Updated distance dictionary with weights
        """
        # Small penalty for non-matching edits
        EPSILON = 0.001

        gold_set = deepcopy(gold_edits)
        retdist = deepcopy(dist)

        M = {}  # Edges grouped by position
        G = {}  # Gold edits grouped by position

        # Group edges by their start and end positions
        for edge in E:
            tE = edits[edge]
            s, e = tE[1], tE[2]
            if (s, e) not in M:
                M[(s, e)] = []
            M[(s, e)].append(edge)
            if (s, e) not in G:
                G[(s, e)] = []

        # Group gold edits by their start and end positions
        for gold in gold_set:
            s, e = gold[0], gold[1]
            if (s, e) not in G:
                G[(s, e)] = []
            G[(s, e)].append(gold)

        # Process each group of edges
        for k in sorted(M.keys()):
            M[k] = sorted(M[k])

            # Special handling for insertion case (start == end)
            if k[0] == k[1]:  # insertion case
                lptr = 0
                rptr = len(M[k]) - 1
                cur = lptr

                g_lptr = 0
                g_rptr = len(G[k]) - 1

                while lptr <= rptr:
                    hasGoldMatch = False
                    edge = M[k][cur]
                    thisEdit = edits[edge]

                    # Determine which gold edits to check
                    if cur == lptr:
                        cur_gold = list(range(g_lptr, g_rptr + 1))
                    else:
                        cur_gold = reversed(list(range(g_lptr, g_rptr + 1)))

                    # Check if current edit matches any gold edit
                    for i in cur_gold:
                        gold = G[k][i]
                        if (
                            thisEdit[1] == gold[0]
                            and thisEdit[2] == gold[1]
                            and thisEdit[3] == gold[2]
                            and thisEdit[4] in gold[3]
                        ):
                            hasGoldMatch = True
                            # Strongly prefer matching edits
                            retdist[edge] = -len(E)
                            if cur == lptr:
                                # g_lptr += 1 # why?
                                g_lptr = i + 1
                            else:
                                # g_rptr -= 1 # why?
                                g_rptr = i - 1
                            break

                    # Penalize non-matching edits
                    if not hasGoldMatch and thisEdit[0] != "noop":
                        retdist[edge] += EPSILON

                    # Update pointers based on match result
                    if hasGoldMatch:
                        if cur == lptr:
                            lptr += 1
                            while lptr < len(M[k]) and M[k][lptr][0] != M[k][cur][1]:
                                if edits[M[k][lptr]] != "noop":
                                    retdist[M[k][lptr]] += EPSILON
                                lptr += 1
                            cur = lptr
                        else:
                            rptr -= 1
                            while rptr >= 0 and M[k][rptr][1] != M[k][cur][0]:
                                if edits[M[k][rptr]] != "noop":
                                    retdist[M[k][rptr]] += EPSILON
                                rptr -= 1
                            cur = rptr
                    else:
                        if cur == lptr:
                            lptr += 1
                            cur = rptr
                        else:
                            rptr -= 1
                            cur = lptr
            else:
                # For deletion or substitution, check all edges against all gold edits
                for edge in M[k]:
                    hasGoldMatch = False
                    thisEdit = edits[edge]
                    for gold in G[k]:
                        if (
                            thisEdit[1] == gold[0]
                            and thisEdit[2] == gold[1]
                            and thisEdit[3] == gold[2]
                            and thisEdit[4] in gold[3]
                        ):
                            hasGoldMatch = True
                            # Strongly prefer matching edits
                            retdist[edge] = -len(E)
                            break
                    if not hasGoldMatch and thisEdit[0] != "noop":
                        # Penalize non-matching edits
                        retdist[edge] += EPSILON
        return retdist

    def best_edit_seq_bf(self, V, E, dist, edits):
        """Finds the best edit sequence using the Bellman-Ford algorithm.

        Computes shortest paths from (0,0) to all vertices, then backtracks
        from the end vertex to get the edit sequence.

        Args:
            V (List): List of vertices
            E (List): List of edges
            dist (Dict): Dictionary of distances/weights
            edits (Dict): Dictionary of edit operations

        Returns:
            List: Best edit sequence
        """
        # Initialize distances
        thisdist = {}
        path = {}
        for v in V:
            thisdist[v] = float("inf")
        thisdist[(0, 0)] = 0

        # Bellman-Ford algorithm to find shortest paths
        for i in range(len(V) - 1):
            for edge in E:
                v = edge[0]  # Source vertex
                w = edge[1]  # Destination vertex
                if thisdist[v] + dist[edge] < thisdist[w]:
                    thisdist[w] = thisdist[v] + dist[edge]
                    path[w] = v

        # Backtrack to get edit sequence
        v = sorted(V)[-1]  # Start from end vertex
        editSeq = []
        while True:
            try:
                # Previous vertex
                w = path[v]
            except KeyError:
                break
            edit = edits[(w, v)]
            if edit[0] != "noop":
                # Skip no-operation edits
                editSeq.append((edit[1], edit[2], edit[3], edit[4]))
            v = w
        return editSeq

    def merge_graph(self, V1: List, V2: List, E1: List, E2: List, dist1: Dict, dist2: Dict, edits1: Dict, edits2: Dict):
        """Merges two edit graphs into a single graph.

        Args:
            V1 (List): Vertices from first graph
            V2 (List): Vertices from second graph
            E1 (List): Edges from first graph
            E2 (List): Edges from second graph
            dist1 (Dict): Distances from first graph
            dist2 (Dict): Distances from second graph
            edits1 (Dict): Edits from first graph
            edits2 (Dict): Edits from second graph

        Returns:
            Tuple: (V, E, dist, edits) merged graph
        """
        # Merge vertices
        V = deepcopy(V1)
        for v in V2:
            if v not in V:
                V.append(v)
        V = sorted(V)

        # Merge edges
        E = E1
        for e in E2:
            if e not in V:
                E.append(e)
        E = sorted(E)

        # Merge distances
        dist = deepcopy(dist1)
        for k in list(dist2.keys()):
            if k not in list(dist.keys()):
                dist[k] = dist2[k]
            else:
                if dist[k] != dist2[k]:
                    print("WARNING: merge_graph: distance does not match!", file=sys.stderr)
                    dist[k] = min(dist[k], dist2[k])

        # edit contents
        edits = deepcopy(edits1)
        for e in list(edits2.keys()):
            if e not in list(edits.keys()):
                edits[e] = edits2[e]
            else:
                if edits[e] != edits2[e]:
                    print("WARNING: merge_graph: edit does not match!", file=sys.stderr)
        return (V, E, dist, edits)

    def matchSeq(self, editSeq: List, gold_edits: List):
        """Matches an edit sequence against gold standard edits.

        Finds the maximum number of matching edits between the candidate edit sequence
        and gold standard edits, preserving the order of edits.

        Args:
            editSeq (List): Candidate edit sequence
            gold_edits (List): Gold standard edit sequence

        Returns:
            List: Subset of candidate edits that match gold standard edits
        """
        m = []
        goldSeq = deepcopy(gold_edits)
        # Track position in gold sequence to maintain order
        last_index = 0

        # Process edits in reverse order to handle overlapping edits correctly
        for e in reversed(editSeq):
            # Search for a matching gold edit starting from last matched position
            for i in range(last_index, len(goldSeq)):
                g = goldSeq[i]
                if self.matchEdit(e, g):
                    m.append(e)
                    last_index = i + 1
                    break
        return m

    def matchEdit(self, e, g):
        """Determines if a candidate edit matches a gold standard edit.

        An edit matches if all components (start offset, end offset,
        original string, and correction string) match exactly.

        Args:
            e (Tuple): Candidate edit (start_offset, end_offset, original_string, correction_string)
            g (Tuple): Gold standard edit (start_offset, end_offset, original_string, [correction_strings])

        Returns:
            bool: True if edits match, False otherwise
        """
        # Check start offset
        if e[0] != g[0]:
            return False
        # Check end offset
        if e[1] != g[1]:
            return False
        # Check original string
        if e[2] != g[2]:
            return False
        # Check correction string (gold may have multiple acceptable corrections)
        if not e[3] in g[3]:
            return False
        # All criteria match
        return True
