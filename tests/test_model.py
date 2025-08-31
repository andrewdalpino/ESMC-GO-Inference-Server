import unittest
from unittest.mock import patch, MagicMock, Mock
import torch
import networkx as nx
import sys
import os

# Add the app directory to the path so we can import the model module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.model import GoTermClassifier


class TestGoTermClassifier(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock DiGraph for testing
        self.mock_graph = nx.DiGraph()
        self.mock_graph.add_node("GO:0000001", name="test term 1")
        self.mock_graph.add_node("GO:0000002", name="test term 2")
        self.mock_graph.add_edge("GO:0000001", "GO:0000002")

        # Mock model parameters
        self.model_name = "andrewdalpino/ESMC-300M-Protein-Function"
        self.context_length = 1024
        self.quantize = False
        self.quant_group_size = 192
        self.device = "cpu"

        # Sample protein sequence for testing
        self.test_sequence = "MAEGEITTFTALTEKFNLPPGNYKKPKLLYCSNGGHFLRILPDGTVDGTRDRSDQHIQLQLSAESVGEVYIKSTETGQYLAMDTSGLLYGSQTPNEECLFLERLEENHYNTYTSKKHAEKNWFVGLKKNGSCKRGPRTHYGQKAILFLPLPV"

    @patch("app.model.EsmcGoTermClassifier")
    @patch("app.model.EsmSequenceTokenizer")
    @patch("app.model.torch")
    @patch("app.model.cuda_is_available")
    @patch("app.model.is_bf16_supported")
    def test_init_successful(
        self, mock_bf16, mock_cuda, mock_torch, mock_tokenizer_class, mock_model_class
    ):
        """Test successful initialization of GoTermClassifier."""
        # Configure mocks
        mock_cuda.return_value = True
        mock_bf16.return_value = True
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.return_value = mock_tokenizer
        mock_torch.compile.return_value = mock_model

        # Initialize classifier
        classifier = GoTermClassifier(
            name=self.model_name,
            graph=self.mock_graph,
            context_length=self.context_length,
            quantize=self.quantize,
            quant_group_size=self.quant_group_size,
            device=self.device,
        )

        # Verify correct initialization
        self.assertEqual(classifier.tokenizer, mock_tokenizer)
        self.assertEqual(classifier.model, mock_model)
        self.assertEqual(classifier.context_length, self.context_length)
        self.assertEqual(classifier.device, self.device)

        # Verify method calls
        mock_model_class.from_pretrained.assert_called_once_with(self.model_name)
        mock_model.to.assert_called_once()
        mock_model.load_gene_ontology.assert_called_once_with(self.mock_graph)
        mock_model.eval.assert_called_once()

        # Verify quantize not called since quantize=False
        mock_model.quantize_weights.assert_not_called()

    @patch("app.model.EsmcGoTermClassifier")
    @patch("app.model.EsmSequenceTokenizer")
    @patch("app.model.torch")
    @patch("app.model.cuda_is_available")
    @patch("app.model.is_bf16_supported")
    def test_init_with_quantization(
        self, mock_bf16, mock_cuda, mock_torch, mock_tokenizer_class, mock_model_class
    ):
        """Test initialization with quantization enabled."""
        # Configure mocks
        mock_cuda.return_value = True
        mock_bf16.return_value = True
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.return_value = mock_tokenizer
        mock_torch.compile.return_value = mock_model

        # Initialize classifier with quantization
        classifier = GoTermClassifier(
            name=self.model_name,
            graph=self.mock_graph,
            context_length=self.context_length,
            quantize=True,
            quant_group_size=self.quant_group_size,
            device=self.device,
        )

        # Verify quantize called with correct parameters
        mock_model.quantize_weights.assert_called_once_with(
            group_size=self.quant_group_size
        )

    @patch("app.model.EsmcGoTermClassifier")
    @patch("app.model.EsmSequenceTokenizer")
    @patch("app.model.torch")
    @patch("app.model.cuda_is_available")
    def test_init_invalid_model(
        self, mock_cuda, mock_torch, mock_tokenizer_class, mock_model_class
    ):
        """Test initialization with invalid model name."""
        mock_cuda.return_value = True

        # Test with invalid model name
        with self.assertRaises(ValueError) as context:
            GoTermClassifier(
                name="invalid_model_name",
                graph=self.mock_graph,
                context_length=self.context_length,
                quantize=self.quantize,
                quant_group_size=self.quant_group_size,
                device=self.device,
            )

        self.assertIn(
            "Model invalid_model_name is not available", str(context.exception)
        )

    @patch("app.model.EsmcGoTermClassifier")
    @patch("app.model.EsmSequenceTokenizer")
    @patch("app.model.torch")
    @patch("app.model.cuda_is_available")
    def test_init_invalid_context_length(
        self, mock_cuda, mock_torch, mock_tokenizer_class, mock_model_class
    ):
        """Test initialization with invalid context length."""
        mock_cuda.return_value = True

        # Test with invalid context length
        with self.assertRaises(ValueError) as context:
            GoTermClassifier(
                name=self.model_name,
                graph=self.mock_graph,
                context_length=0,
                quantize=self.quantize,
                quant_group_size=self.quant_group_size,
                device=self.device,
            )

        self.assertIn("Context length must be greater than 0", str(context.exception))

    @patch("app.model.EsmcGoTermClassifier")
    @patch("app.model.EsmSequenceTokenizer")
    @patch("app.model.torch")
    @patch("app.model.cuda_is_available")
    def test_init_cuda_not_available(
        self, mock_cuda, mock_torch, mock_tokenizer_class, mock_model_class
    ):
        """Test initialization with CUDA requested but not available."""
        mock_cuda.return_value = False

        # Test with CUDA requested but not available
        with self.assertRaises(ValueError) as context:
            GoTermClassifier(
                name=self.model_name,
                graph=self.mock_graph,
                context_length=self.context_length,
                quantize=self.quantize,
                quant_group_size=self.quant_group_size,
                device="cuda",
            )

        self.assertIn("CUDA is not supported on this device", str(context.exception))

    @patch("app.model.EsmcGoTermClassifier")
    @patch("app.model.EsmSequenceTokenizer")
    @patch("app.model.torch")
    @patch("app.model.cuda_is_available")
    @patch("app.model.is_bf16_supported")
    def test_predict_terms(
        self, mock_bf16, mock_cuda, mock_torch, mock_tokenizer_class, mock_model_class
    ):
        """Test predict_terms method."""
        # Configure mocks
        mock_cuda.return_value = True
        mock_bf16.return_value = True
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.return_value = mock_tokenizer
        mock_torch.compile.return_value = mock_model

        # Mock tokenizer output
        mock_tokenizer.return_value = {"input_ids": [1, 2, 3, 4, 5]}

        # Mock model predict_terms output
        expected_probabilities = {
            "GO:0000001": 0.9,
            "GO:0000002": 0.8,
        }
        mock_model.predict_terms.return_value = expected_probabilities

        # Initialize classifier
        classifier = GoTermClassifier(
            name=self.model_name,
            graph=self.mock_graph,
            context_length=self.context_length,
            quantize=self.quantize,
            quant_group_size=self.quant_group_size,
            device=self.device,
        )

        # Test predict_terms
        result = classifier.predict_terms(self.test_sequence, top_p=0.5)

        # Verify result
        self.assertEqual(result, expected_probabilities)

        # Verify tokenizer was called correctly
        mock_tokenizer.assert_called_once_with(
            self.test_sequence,
            max_length=self.context_length,
            truncation=True,
        )

        # Verify model predict_terms was called correctly
        mock_model.predict_terms.assert_called_once()

    @patch("app.model.EsmcGoTermClassifier")
    @patch("app.model.EsmSequenceTokenizer")
    @patch("app.model.torch")
    @patch("app.model.cuda_is_available")
    @patch("app.model.is_bf16_supported")
    def test_predict_subgraph(
        self, mock_bf16, mock_cuda, mock_torch, mock_tokenizer_class, mock_model_class
    ):
        """Test predict_subgraph method."""
        # Configure mocks
        mock_cuda.return_value = True
        mock_bf16.return_value = True
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.return_value = mock_tokenizer
        mock_torch.compile.return_value = mock_model

        # Mock tokenizer output
        mock_tokenizer.return_value = {"input_ids": [1, 2, 3, 4, 5]}

        # Create a mock subgraph and probabilities
        mock_subgraph = nx.DiGraph()
        mock_subgraph.add_node("GO:0000001", name="test term 1")
        mock_subgraph.add_node("GO:0000002", name="test term 2")
        mock_subgraph.add_edge("GO:0000001", "GO:0000002")

        expected_probabilities = {
            "GO:0000001": 0.9,
            "GO:0000002": 0.8,
        }

        # Mock model predict_subgraph output
        mock_model.predict_subgraph.return_value = (
            mock_subgraph,
            expected_probabilities,
        )

        # Initialize classifier
        classifier = GoTermClassifier(
            name=self.model_name,
            graph=self.mock_graph,
            context_length=self.context_length,
            quantize=self.quantize,
            quant_group_size=self.quant_group_size,
            device=self.device,
        )

        # Test predict_subgraph
        result_subgraph, result_probabilities = classifier.predict_subgraph(
            self.test_sequence, top_p=0.5
        )

        # Verify result
        self.assertEqual(result_subgraph, mock_subgraph)
        self.assertEqual(result_probabilities, expected_probabilities)

        # Verify tokenizer was called correctly
        mock_tokenizer.assert_called_once_with(
            self.test_sequence,
            max_length=self.context_length,
            truncation=True,
        )

        # Verify model predict_subgraph was called correctly
        mock_model.predict_subgraph.assert_called_once()


if __name__ == "__main__":
    unittest.main()
