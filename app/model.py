import torch

from esm.tokenization import EsmSequenceTokenizer

from esmc_function_classifier.model import EsmcGoTermClassifier

from networkx import DiGraph

import networkx as nx


class GoTermClassifier:
    AVAILABLE_MODELS = {
        "andrewdalpino/ESMC-300M-Protein-Function",
        "andrewdalpino/ESMC-600M-Protein-Function",
    }

    def __init__(
        self,
        model_name: str,
        graph: DiGraph,
        context_length: int,
        device: str,
        dtype: torch.dtype,
    ):
        """
        Args:
            model_name: HuggingFace model identifier for the ESMC model.
            context_length: Maximum length of the input sequence.
            device: Device to run the model on (e.g., "cuda" or "cpu").
        """

        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Model {model_name} is not available. "
                f"Available models: {self.AVAILABLE_MODELS}"
            )

        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError(
                "The provided gene ontology must be a directed acyclic graph (DAG)."
            )

        if context_length <= 0:
            raise ValueError("Context length must be greater than 0.")

        tokenizer = EsmSequenceTokenizer()

        model = EsmcGoTermClassifier.from_pretrained(model_name)

        model = model.to(device=device, dtype=dtype)

        model.eval()

        model.load_gene_ontology(graph)

        self.tokenizer = tokenizer
        self.model = model
        self.context_length = context_length
        self.device = device

    @torch.no_grad()
    def predict_terms(self, sequence: str, top_p: float = 0.5) -> dict[str, float]:
        """Get the GO term probabilities for a given protein sequence."""

        out = self.tokenizer(
            sequence,
            max_length=self.context_length,
            truncation=True,
        )

        input_ids = torch.tensor(out["input_ids"], dtype=torch.int64).to(self.device)

        go_term_probabilities = self.model.predict_terms(input_ids, top_p)

        return go_term_probabilities

    def predict_subgraph(
        self, sequence: str, top_p: float = 0.5
    ) -> tuple[DiGraph, dict[str, float]]:
        """Get the GO subgraph for a given protein sequence."""

        out = self.tokenizer(
            sequence,
            max_length=self.context_length,
            truncation=True,
        )

        input_ids = torch.tensor(out["input_ids"], dtype=torch.int64).to(self.device)

        subgraph, go_term_probabilities = self.model.predict_subgraph(input_ids, top_p)

        return subgraph, go_term_probabilities
