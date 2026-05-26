import unittest
from unittest.mock import MagicMock, PropertyMock, patch, call

import torch
from networkx import DiGraph

from app.model import GoTermClassifier


class TestGoTermClassifierConstructor(unittest.TestCase):
    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_model.to.return_value = self.mock_model
        self.mock_model.tokenizer = MagicMock()

        self.graph = DiGraph()

    def test_invalid_model_name_raises_value_error(self):
        with (
            patch("app.model.ESMCProteinFunction.from_pretrained"),
            patch("app.model.Semaphore"),
        ):
            with self.assertRaises(ValueError) as ctx:
                GoTermClassifier(
                    name="invalid/model",
                    graph=self.graph,
                    context_length=128,
                    device="cpu",
                    quantize=False,
                    quant_group_size=64,
                    max_concurrency=1,
                )

        self.assertIn("invalid/model", str(ctx.exception))

    def test_context_length_zero_raises_value_error(self):
        with (
            patch("app.model.ESMCProteinFunction.from_pretrained"),
            patch("app.model.Semaphore"),
        ):
            with self.assertRaises(ValueError) as ctx:
                GoTermClassifier(
                    name="andrewdalpino/ESMC-Protein-Function-V1-300M",
                    graph=self.graph,
                    context_length=0,
                    device="cpu",
                    quantize=False,
                    quant_group_size=64,
                    max_concurrency=1,
                )

        self.assertIn("Context length", str(ctx.exception))

    def test_context_length_negative_raises_value_error(self):
        with (
            patch("app.model.ESMCProteinFunction.from_pretrained"),
            patch("app.model.Semaphore"),
        ):
            with self.assertRaises(ValueError) as ctx:
                GoTermClassifier(
                    name="andrewdalpino/ESMC-Protein-Function-V1-300M",
                    graph=self.graph,
                    context_length=-10,
                    device="cpu",
                    quantize=False,
                    quant_group_size=64,
                    max_concurrency=1,
                )

        self.assertIn("Context length", str(ctx.exception))

    @patch("app.model.cuda_is_available", return_value=False)
    def test_cuda_device_when_not_available_raises_value_error(self, _mock_cuda):
        with (
            patch("app.model.ESMCProteinFunction.from_pretrained"),
            patch("app.model.Semaphore"),
        ):
            with self.assertRaises(ValueError) as ctx:
                GoTermClassifier(
                    name="andrewdalpino/ESMC-Protein-Function-V1-300M",
                    graph=self.graph,
                    context_length=128,
                    device="cuda",
                    quantize=False,
                    quant_group_size=64,
                    max_concurrency=1,
                )

        self.assertIn("CUDA", str(ctx.exception))

    @patch("app.model.mps_is_available", return_value=False)
    def test_mps_device_when_not_available_raises_value_error(self, _mock_mps):
        with (
            patch("app.model.ESMCProteinFunction.from_pretrained"),
            patch("app.model.Semaphore"),
        ):
            with self.assertRaises(ValueError) as ctx:
                GoTermClassifier(
                    name="andrewdalpino/ESMC-Protein-Function-V1-300M",
                    graph=self.graph,
                    context_length=128,
                    device="mps",
                    quantize=False,
                    quant_group_size=64,
                    max_concurrency=1,
                )

        self.assertIn("MPS", str(ctx.exception))

    def test_max_concurrency_zero_raises_value_error(self):
        with (
            patch("app.model.ESMCProteinFunction.from_pretrained"),
            patch("app.model.Semaphore"),
        ):
            with self.assertRaises(ValueError) as ctx:
                GoTermClassifier(
                    name="andrewdalpino/ESMC-Protein-Function-V1-300M",
                    graph=self.graph,
                    context_length=128,
                    device="cpu",
                    quantize=False,
                    quant_group_size=64,
                    max_concurrency=0,
                )

        self.assertIn("Max concurrency", str(ctx.exception))

    def test_max_concurrency_negative_raises_value_error(self):
        with (
            patch("app.model.ESMCProteinFunction.from_pretrained"),
            patch("app.model.Semaphore"),
        ):
            with self.assertRaises(ValueError) as ctx:
                GoTermClassifier(
                    name="andrewdalpino/ESMC-Protein-Function-V1-300M",
                    graph=self.graph,
                    context_length=128,
                    device="cpu",
                    quantize=False,
                    quant_group_size=64,
                    max_concurrency=-1,
                )

        self.assertIn("Max concurrency", str(ctx.exception))

    @patch("app.model.Semaphore")
    def test_happy_path_no_quantize_cpu(self, mock_semaphore):
        mock_from_pretrained = patch(
            "app.model.ESMCProteinFunction.from_pretrained",
            return_value=self.mock_model,
        )
        mock_from_pretrained.start()
        self.addCleanup(mock_from_pretrained.stop)

        classifier = GoTermClassifier(
            name="andrewdalpino/ESMC-Protein-Function-V1-300M",
            graph=self.graph,
            context_length=128,
            device="cpu",
            quantize=False,
            quant_group_size=64,
            max_concurrency=2,
        )

        self.assertEqual(classifier.name, "andrewdalpino/ESMC-Protein-Function-V1-300M")
        self.assertEqual(classifier.context_length, 128)
        self.assertEqual(classifier.device, "cpu")
        self.assertFalse(classifier.quantize)
        self.assertEqual(classifier.quant_group_size, 64)
        self.assertEqual(classifier.max_concurrency, 2)
        self.assertEqual(classifier.model, self.mock_model)

        mock_semaphore.assert_called_once_with(2)
        self.assertEqual(classifier.limiter, mock_semaphore.return_value)

        self.mock_model.to.assert_has_calls([
            call(dtype=torch.float16),
            call("cpu"),
        ])
        self.mock_model.load_gene_ontology.assert_called_once_with(self.graph)
        self.mock_model.eval.assert_called_once()

    @patch("app.model.Semaphore")
    @patch("app.model.cuda_is_available", return_value=True)
    @patch("app.model.is_bf16_supported", return_value=True)
    def test_happy_path_no_quantize_cuda_bf16(
        self, _mock_bf16, _mock_cuda, mock_semaphore
    ):
        mock_from_pretrained = patch(
            "app.model.ESMCProteinFunction.from_pretrained",
            return_value=self.mock_model,
        )
        mock_from_pretrained.start()
        self.addCleanup(mock_from_pretrained.stop)

        classifier = GoTermClassifier(
            name="andrewdalpino/ESMC-Protein-Function-V1-300M",
            graph=self.graph,
            context_length=128,
            device="cuda",
            quantize=False,
            quant_group_size=64,
            max_concurrency=1,
        )

        self.assertEqual(classifier.device, "cuda")

        self.mock_model.to.assert_has_calls([
            call(dtype=torch.bfloat16),
            call("cuda"),
        ])

    @patch("app.model.Semaphore")
    @patch("app.model.cuda_is_available", return_value=True)
    @patch("app.model.is_bf16_supported", return_value=False)
    def test_happy_path_no_quantize_cuda_float16(
        self, _mock_bf16, _mock_cuda, mock_semaphore
    ):
        mock_from_pretrained = patch(
            "app.model.ESMCProteinFunction.from_pretrained",
            return_value=self.mock_model,
        )
        mock_from_pretrained.start()
        self.addCleanup(mock_from_pretrained.stop)

        GoTermClassifier(
            name="andrewdalpino/ESMC-Protein-Function-V1-300M",
            graph=self.graph,
            context_length=128,
            device="cuda",
            quantize=False,
            quant_group_size=64,
            max_concurrency=1,
        )

        self.mock_model.to.assert_has_calls([
            call(dtype=torch.float16),
            call("cuda"),
        ])

    @patch("app.model.Semaphore")
    def test_happy_path_quantize(self, mock_semaphore):
        mock_from_pretrained = patch(
            "app.model.ESMCProteinFunction.from_pretrained",
            return_value=self.mock_model,
        )
        mock_from_pretrained.start()
        self.addCleanup(mock_from_pretrained.stop)

        classifier = GoTermClassifier(
            name="andrewdalpino/ESMC-Protein-Function-V1-300M",
            graph=self.graph,
            context_length=128,
            device="cpu",
            quantize=True,
            quant_group_size=32,
            max_concurrency=1,
        )

        self.assertTrue(classifier.quantize)
        self.assertEqual(classifier.quant_group_size, 32)

        self.mock_model.quantize_weights.assert_called_once_with(32)
        self.mock_model.to.assert_called_once_with("cpu")
        self.mock_model.load_gene_ontology.assert_called_once_with(self.graph)
        self.mock_model.eval.assert_called_once()

    def test_happy_path_tokenizer_partial(self):
        mock_tokenizer = MagicMock(return_value={"input_ids": [[1, 2, 3]]})
        self.mock_model.tokenizer = mock_tokenizer

        with (
            patch(
                "app.model.ESMCProteinFunction.from_pretrained",
                return_value=self.mock_model,
            ),
            patch("app.model.Semaphore"),
        ):
            classifier = GoTermClassifier(
                name="andrewdalpino/ESMC-Protein-Function-V1-300M",
                graph=self.graph,
                context_length=128,
                device="cpu",
                quantize=False,
                quant_group_size=64,
                max_concurrency=1,
            )

        result = classifier.tokenize(["MKLL"])

        mock_tokenizer.assert_called_once_with(
            ["MKLL"], max_length=128, padding=True, truncation=True
        )
        self.assertEqual(result, {"input_ids": [[1, 2, 3]]})


class TestGoTermClassifierProperty(unittest.TestCase):
    def test_num_parameters_delegates_to_model(self):
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        type(mock_model).num_params = PropertyMock(return_value=300_000_000)

        with (
            patch(
                "app.model.ESMCProteinFunction.from_pretrained",
                return_value=mock_model,
            ),
            patch("app.model.Semaphore"),
        ):
            classifier = GoTermClassifier(
                name="andrewdalpino/ESMC-Protein-Function-V1-300M",
                graph=DiGraph(),
                context_length=128,
                device="cpu",
                quantize=False,
                quant_group_size=64,
                max_concurrency=1,
            )

        self.assertEqual(classifier.num_parameters, 300_000_000)


class TestGoTermClassifierPrediction(unittest.TestCase):
    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_model.to.return_value = self.mock_model
        self.mock_model.tokenizer = MagicMock(
            return_value={"input_ids": [[1, 2, 3]]}
        )

        self.mock_tensor = MagicMock()
        self.mock_tensor_dev = MagicMock()
        self.mock_tensor.to.return_value = self.mock_tensor_dev

        self.mock_semaphore = MagicMock()

        self.patchers = [
            patch(
                "app.model.ESMCProteinFunction.from_pretrained",
                return_value=self.mock_model,
            ),
            patch("app.model.Semaphore", return_value=self.mock_semaphore),
            patch("app.model.torch.tensor", return_value=self.mock_tensor),
        ]
        for p in self.patchers:
            p.start()

        self.classifier = GoTermClassifier(
            name="andrewdalpino/ESMC-Protein-Function-V1-300M",
            graph=DiGraph(),
            context_length=128,
            device="cpu",
            quantize=False,
            quant_group_size=64,
            max_concurrency=1,
        )

    def tearDown(self):
        for p in reversed(self.patchers):
            p.stop()

    def test_predict_mf_terms(self):
        expected = [{"GO:0001": 0.95}]
        self.mock_model.predict_mf_terms.return_value = expected

        result = self.classifier.predict_mf_terms(["MKLL"], top_p=0.5)

        self.mock_model.tokenizer.assert_called_once_with(
            ["MKLL"], max_length=128, padding=True, truncation=True
        )
        self.mock_tensor.to.assert_called_once_with("cpu")
        self.mock_model.predict_mf_terms.assert_called_once_with(
            self.mock_tensor_dev, 0.5
        )
        self.assertEqual(result, expected)

    def test_predict_bp_terms(self):
        expected = [{"GO:0002": 0.87}]
        self.mock_model.predict_bp_terms.return_value = expected

        result = self.classifier.predict_bp_terms(["MKLL"], top_p=0.3)

        self.mock_model.tokenizer.assert_called_once()
        self.mock_model.predict_bp_terms.assert_called_once_with(
            self.mock_tensor_dev, 0.3
        )
        self.assertEqual(result, expected)

    def test_predict_cc_terms(self):
        expected = [{"GO:0003": 0.76}]
        self.mock_model.predict_cc_terms.return_value = expected

        result = self.classifier.predict_cc_terms(["MKLL"], top_p=0.9)

        self.mock_model.tokenizer.assert_called_once()
        self.mock_model.predict_cc_terms.assert_called_once_with(
            self.mock_tensor_dev, 0.9
        )
        self.assertEqual(result, expected)

    def test_predict_all_terms(self):
        expected_mf = [{"GO:0001": 0.95}]
        expected_bp = [{"GO:0002": 0.87}]
        expected_cc = [{"GO:0003": 0.76}]
        self.mock_model.predict_all_terms.return_value = (
            expected_mf, expected_bp, expected_cc
        )

        mf, bp, cc = self.classifier.predict_all_terms(["MKLL"], top_p=0.5)

        self.mock_model.tokenizer.assert_called_once()
        self.mock_model.predict_all_terms.assert_called_once_with(
            self.mock_tensor_dev, 0.5
        )
        self.assertEqual(mf, expected_mf)
        self.assertEqual(bp, expected_bp)
        self.assertEqual(cc, expected_cc)

    def test_predict_mf_subgraphs(self):
        expected = [MagicMock(spec=DiGraph)]
        self.mock_model.predict_mf_subgraphs.return_value = expected

        result = self.classifier.predict_mf_subgraphs(["MKLL"], top_p=0.5)

        self.mock_model.tokenizer.assert_called_once()
        self.mock_model.predict_mf_subgraphs.assert_called_once_with(
            self.mock_tensor_dev, 0.5
        )
        self.assertEqual(result, expected)

    def test_predict_bp_subgraphs(self):
        expected = [MagicMock(spec=DiGraph)]
        self.mock_model.predict_bp_subgraphs.return_value = expected

        result = self.classifier.predict_bp_subgraphs(["MKLL"], top_p=0.5)

        self.mock_model.tokenizer.assert_called_once()
        self.mock_model.predict_bp_subgraphs.assert_called_once_with(
            self.mock_tensor_dev, 0.5
        )
        self.assertEqual(result, expected)

    def test_predict_cc_subgraphs(self):
        expected = [MagicMock(spec=DiGraph)]
        self.mock_model.predict_cc_subgraphs.return_value = expected

        result = self.classifier.predict_cc_subgraphs(["MKLL"], top_p=0.5)

        self.mock_model.tokenizer.assert_called_once()
        self.mock_model.predict_cc_subgraphs.assert_called_once_with(
            self.mock_tensor_dev, 0.5
        )
        self.assertEqual(result, expected)

    def test_predict_all_subgraphs(self):
        expected_mf = [MagicMock(spec=DiGraph)]
        expected_bp = [MagicMock(spec=DiGraph)]
        expected_cc = [MagicMock(spec=DiGraph)]
        self.mock_model.predict_all_subgraphs.return_value = (
            expected_mf, expected_bp, expected_cc
        )

        mf, bp, cc = self.classifier.predict_all_subgraphs(["MKLL"], top_p=0.5)

        self.mock_model.tokenizer.assert_called_once()
        self.mock_model.predict_all_subgraphs.assert_called_once_with(
            self.mock_tensor_dev, 0.5
        )
        self.assertEqual(mf, expected_mf)
        self.assertEqual(bp, expected_bp)
        self.assertEqual(cc, expected_cc)
