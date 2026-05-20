from functools import partial

import torch

from torch.cuda import is_available as cuda_is_available, is_bf16_supported
from torch.backends.mps import is_available as mps_is_available

from esmc_protein_function.model import ESMCProteinFunction

from networkx import DiGraph

from threading import Semaphore


class GoTermClassifier:
    AVAILABLE_MODELS = {
        "andrewdalpino/ESMC-Protein-Function-V1-300M",
        "andrewdalpino/ESMC-Protein-Function-V1-600M",
    }

    def __init__(
        self,
        name: str,
        graph: DiGraph,
        context_length: int,
        device: str,
        quantize: bool,
        quant_group_size: int,
        max_concurrency: int,
    ):
        """
        Args:
            name: HuggingFace model identifier for the model.
            graph: A NetworkX DiGraph representing the Gene Ontology DAG.
            context_length: Maximum length of the input sequence.
            device: Device to run the model on (e.g., "cuda" or "cpu").
            quantize: Whether to quantize the model weights.
            quant_group_size: Group size for quantization.
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

        model = ESMCProteinFunction.from_pretrained(name)

        if quantize:
            model.quantize_weights(quant_group_size)
        else:
            dtype = (
                torch.bfloat16
                if "cuda" in device and is_bf16_supported()
                else torch.float16
            )

            model = model.to(dtype=dtype)

        model = model.to(device)

        model.load_gene_ontology(graph)

        model.eval()

        limiter = Semaphore(max_concurrency)

        tokenize = partial(
            model.tokenizer,
            max_length=context_length,
            padding=True,
            truncation=True,
        )

        self.name = name
        self.model = model
        self.context_length = context_length
        self.device = device
        self.quantize = quantize
        self.quant_group_size = quant_group_size
        self.max_concurrency = max_concurrency
        self.limiter = limiter
        self.tokenize = tokenize

    @property
    def num_parameters(self) -> int:
        """Return the number of parameters in the model."""

        return self.model.num_params

    def predict_mf_terms(
        self, sequences: list[str], top_p: float
    ) -> list[dict[str, float]]:
        """Predict the MF GO term probabilities for a given protein sequence."""

        out = self.tokenize(sequences)

        input_ids = torch.tensor(out["input_ids"], dtype=torch.int32)

        with self.limiter:
            input_ids = input_ids.to(self.device)

            terms = self.model.predict_mf_terms(input_ids, top_p)

        return terms

    def predict_bp_terms(
        self, sequences: list[str], top_p: float
    ) -> list[dict[str, float]]:
        """Predict the BP GO term probabilities for a given protein sequence."""

        out = self.tokenize(sequences)

        input_ids = torch.tensor(out["input_ids"], dtype=torch.int32)

        with self.limiter:
            input_ids = input_ids.to(self.device)

            terms = self.model.predict_bp_terms(input_ids, top_p)

        return terms

    def predict_cc_terms(
        self, sequences: list[str], top_p: float
    ) -> list[dict[str, float]]:
        """Predict the CC GO term probabilities for a given protein sequence."""

        out = self.tokenize(sequences)

        input_ids = torch.tensor(out["input_ids"], dtype=torch.int32)

        with self.limiter:
            input_ids = input_ids.to(self.device)

            terms = self.model.predict_cc_terms(input_ids, top_p)

        return terms

    def predict_all_terms(
        self, sequences: list[str], top_p: float
    ) -> tuple[list[dict[str, float]], ...]:
        """Predict all the GO term probabilities for a given protein sequence."""

        out = self.tokenize(sequences)

        input_ids = torch.tensor(out["input_ids"], dtype=torch.int32)

        with self.limiter:
            input_ids = input_ids.to(self.device)

            mf_terms, bp_terms, cc_terms = self.model.predict_all_terms(
                input_ids, top_p
            )

        return mf_terms, bp_terms, cc_terms

    def predict_mf_subgraphs(
        self, sequences: list[str], top_p: float
    ) -> tuple[list[DiGraph], list[dict[str, float]]]:
        """Predict the GO MF subgraphs for a given protein sequence."""

        out = self.tokenize(sequences)

        input_ids = torch.tensor(out["input_ids"], dtype=torch.int32)

        with self.limiter:
            input_ids = input_ids.to(self.device)

            subgraphs, terms = self.model.predict_mf_subgraphs(input_ids, top_p)

        return subgraphs, terms

    def predict_bp_subgraphs(
        self, sequences: list[str], top_p: float
    ) -> tuple[list[DiGraph], list[dict[str, float]]]:
        """Predict the GO BP subgraphs for a given protein sequence."""

        out = self.tokenize(sequences)

        input_ids = torch.tensor(out["input_ids"], dtype=torch.int32)

        with self.limiter:
            input_ids = input_ids.to(self.device)

            subgraphs, terms = self.model.predict_bp_subgraphs(input_ids, top_p)

        return subgraphs, terms

    def predict_cc_subgraphs(
        self, sequences: list[str], top_p: float
    ) -> tuple[list[DiGraph], list[dict[str, float]]]:
        """Predict the GO CC subgraphs for a given protein sequence."""

        out = self.tokenize(sequences)

        input_ids = torch.tensor(out["input_ids"], dtype=torch.int32)

        with self.limiter:
            input_ids = input_ids.to(self.device)

            subgraphs, terms = self.model.predict_cc_subgraphs(input_ids, top_p)

        return subgraphs, terms

    def predict_all_subgraphs(
        self, sequences: list[str], top_p: float
    ) -> tuple[tuple[list[DiGraph], list[dict[str, float]]], ...]:
        """Predict all the GO subgraphs for a given protein sequence."""

        out = self.tokenize(sequences)

        input_ids = torch.tensor(out["input_ids"], dtype=torch.int32)

        with self.limiter:
            input_ids = input_ids.to(self.device)

            mf_results, bp_results, cc_results = self.model.predict_all_subgraphs(
                input_ids, top_p
            )

        return mf_results, bp_results, cc_results
