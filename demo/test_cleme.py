import os
import unittest

from cleme.cleme import DependentChunkMetric, IndependentChunkMetric
from cleme.data import M2DataReader


class TestCLEME(unittest.TestCase):
    def setUp(self) -> None:
        self.reader = M2DataReader()
        # Read M2 file
        self.dataset_ref = self.reader.read(f"{os.path.dirname(__file__)}/examples/conll14.errant")
        self.dataset_hyp = self.reader.read(f"{os.path.dirname(__file__)}/examples/conll14-AMU.errant")
        print("Example of reference", self.dataset_ref[-1])
        print("Example of hypothesis", self.dataset_hyp[-1])

    def test_demo(self):
        # Read M2 file
        dataset_ref = self.reader.read(f"{os.path.dirname(__file__)}/examples/demo.errant")
        dataset_hyp = self.reader.read(f"{os.path.dirname(__file__)}/examples/demo-AMU.errant")
        print(len(dataset_ref), len(dataset_hyp))
        print("Example of reference", dataset_ref[-1])
        print("Example of hypothesis", dataset_hyp[-1])

        # Evaluate using CLEME_dependent
        config_dependent = {
            "tp": {"alpha": 2.0, "min_value": 0.75, "max_value": 1.25, "reverse": False},
            "fp": {"alpha": 2.0, "min_value": 0.75, "max_value": 1.25, "reverse": True},
            "fn": {"alpha": 2.0, "min_value": 0.75, "max_value": 1.25, "reverse": False},
        }
        metric_dependent = DependentChunkMetric(weigher_config=config_dependent)
        score, results = metric_dependent.evaluate(dataset_hyp, dataset_ref)
        print(f"==================== Evaluate Demo ====================")
        print(score)

        # Visualize
        metric_dependent.visualize(dataset_ref, dataset_hyp)

    def test_cleme_dependent(self):
        # Read M2 file
        dataset_ref = self.dataset_ref
        dataset_hyp = self.dataset_hyp
        # Evaluate using CLEME_dependent
        config = {
            "tp": {"alpha": 2.0, "min_value": 0.75, "max_value": 1.25, "reverse": False},
            "fp": {"alpha": 2.0, "min_value": 0.75, "max_value": 1.25, "reverse": True},
            "fn": {"alpha": 2.0, "min_value": 0.75, "max_value": 1.25, "reverse": False},
        }
        # No length weighting
        # config = {
        #     "tp": {"alpha": 2.0, "min_value": 1.0, "max_value": 1.0, "reverse": False},
        #     "fp": {"alpha": 2.0, "min_value": 1.0, "max_value": 1.0, "reverse": True},
        #     "fn": {"alpha": 2.0, "min_value": 1.0, "max_value": 1.0, "reverse": False},
        # }
        metric_dependent = DependentChunkMetric(weigher_config=config)
        score, results = metric_dependent.evaluate(dataset_hyp, dataset_ref)
        print(f"==================== Evaluate using CLEME_dependent ====================")
        print(score)

    def test_cleme_independent(self):
        # Read M2 file
        dataset_ref = self.dataset_ref
        dataset_hyp = self.dataset_hyp
        # Evaluate using CLEME_independent
        # config = {
        #     "tp": {"alpha": 2.0, "min_value": 0.75, "max_value": 1.25, "reverse": False},
        #     "fp": {"alpha": 2.0, "min_value": 0.75, "max_value": 1.25, "reverse": True},
        #     "fn": {"alpha": 2.0, "min_value": 0.75, "max_value": 1.25, "reverse": False},
        # }
        # No length weighting
        config = {
            "tp": {"alpha": 2.0, "min_value": 1.0, "max_value": 1.0, "reverse": False},
            "fp": {"alpha": 2.0, "min_value": 1.0, "max_value": 1.0, "reverse": True},
            "fn": {"alpha": 2.0, "min_value": 1.0, "max_value": 1.0, "reverse": False},
        }
        metric_independent = IndependentChunkMetric(weigher_config=config)
        score, results = metric_independent.evaluate(dataset_hyp, dataset_ref)
        print(f"==================== Evaluate using CLEME_independent ====================")
        print(score)

    def test_sentcleme_dependent(self):
        # Read M2 file
        dataset_ref = self.dataset_ref
        dataset_hyp = self.dataset_hyp
        # Evaluate using SentCLEME_dependent
        config = {
            "tp": {"alpha": 10.0, "min_value": 1.00, "max_value": 10.0, "reverse": False},
            "fp": {"alpha": 10.0, "min_value": 0.25, "max_value": 1.00, "reverse": True},
            "fn": {"alpha": 10.0, "min_value": 1.00, "max_value": 1.00, "reverse": False},
        }
        metric_dependent = DependentChunkMetric(scorer="sentence", weigher_config=config)
        score, results = metric_dependent.evaluate(dataset_hyp, dataset_ref)
        print(f"==================== Evaluate using SentCLEME_dependent ====================")
        print(score)

    def test_sentcleme_independent(self):
        # Read M2 file
        dataset_ref = self.dataset_ref
        dataset_hyp = self.dataset_hyp
        # Evaluate using SentCLEME_independent
        config = {
            "tp": {"alpha": 10.0, "min_value": 2.50, "max_value": 10.0, "reverse": False},
            "fp": {"alpha": 10.0, "min_value": 0.25, "max_value": 1.00, "reverse": True},
            "fn": {"alpha": 10.0, "min_value": 1.00, "max_value": 1.00, "reverse": False},
        }
        metric_dependent = IndependentChunkMetric(scorer="sentence", weigher_config=config)
        score, results = metric_dependent.evaluate(dataset_hyp, dataset_ref)
        print(f"==================== Evaluate using SentCLEME_independent ====================")
        print(score)


if __name__ == "__main__":
    tester = TestCLEME()
    tester.setUp()
    tester.test_demo()
    # tester.test_cleme_dependent()
    # tester.test_cleme_independent()
    # tester.test_sentcleme_dependent()
    # tester.test_sentcleme_independent()
