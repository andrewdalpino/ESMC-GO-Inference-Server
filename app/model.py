import torch

from torch.cuda import is_available as cuda_is_available, is_bf16_supported
from torch.backends.mps import is_available as mps_is_available

from torchao.quantization import Int8WeightOnlyConfig, quantize_

from esm.tokenization import EsmSequenceTokenizer

from esmc_function_classifier.model import EsmcGoTermClassifier

from networkx import DiGraph

from threading import Semaphore


class GoTermClassifier:
    AVAILABLE_MODELS = {
        "andrewdalpino/ESMC-300M-Protein-Function",
        "andrewdalpino/ESMC-300M-QAT-Protein-Function",
        "andrewdalpino/ESMC-600M-Protein-Function",
        "andrewdalpino/ESMC-600M-QAT-Protein-Function",
    }

    def __init__(
        self,
        name: str,
        graph: DiGraph,
        context_length: int,
        device: str,
        quantize: bool,
        max_concurrency: int,
    ):
        """
        Args:
            name: HuggingFace model identifier for the ESMC model.
            graph: A NetworkX DiGraph representing the Gene Ontology DAG.
            context_length: Maximum length of the input sequence.
            device: Device to run the model on (e.g., "cuda" or "cpu").
            quantize: Whether to quantize the model weights.
            max_concurrency: Maximum number of concurrent inferences.
        """

        if name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Model {name} is not available. "
                f"Available models: {self.AVAILABLE_MODELS}"
            )

        if context_length <= 0:
            raise ValueError("Context length must be greater than 0.")

        if "cuda" in device and not cuda_is_available():
            raise ValueError("CUDA is not supported on this device.")

        if "mps" in device and not mps_is_available():
            raise ValueError("MPS is not supported on this device.")

        if max_concurrency < 1:
            raise ValueError("Max concurrency must be greater than 0.")

        tokenizer = EsmSequenceTokenizer()

        model = EsmcGoTermClassifier.from_pretrained(name)

        dtype = (
            torch.bfloat16
            if "cuda" in device and is_bf16_supported()
            else torch.float16
        )

        model = model.to(dtype=dtype)

        model = torch.compile(model)

        if quantize:
            quantize_(model, Int8WeightOnlyConfig())

        model = model.to(device)

        model.load_gene_ontology(graph)

        model.eval()

        limiter = Semaphore(max_concurrency)

        self.tokenizer = tokenizer
        self.name = name
        self.model = model
        self.context_length = context_length
        self.device = device
        self.quantize = quantize
        self.max_concurrency = max_concurrency
        self.limiter = limiter

    @property
    def num_parameters(self) -> int:
        """Return the number of parameters in the model."""

        return self.model.num_params

    @torch.inference_mode()
    def predict_terms(self, sequence: str, top_p: float) -> dict[str, float]:
        """Get the GO term probabilities for a given protein sequence."""

        out = self.tokenizer(
            sequence,
            max_length=self.context_length,
            truncation=True,
        )

        input_ids = torch.tensor(out["input_ids"], dtype=torch.int64)

        with self.limiter:
            input_ids = input_ids.to(self.device)

            probabilities = self.model.predict_terms(input_ids, top_p)

        return probabilities

    @torch.inference_mode()
    def predict_subgraph(
        self, sequence: str, top_p: float
    ) -> tuple[DiGraph, dict[str, float]]:
        """Get the GO subgraph for a given protein sequence."""

        out = self.tokenizer(
            sequence,
            max_length=self.context_length,
            truncation=True,
        )

        input_ids = torch.tensor(out["input_ids"], dtype=torch.int64)

        with self.limiter:
            input_ids = input_ids.to(self.device)

            subgraph, probabilities = self.model.predict_subgraph(input_ids, top_p)

        return subgraph, probabilities
