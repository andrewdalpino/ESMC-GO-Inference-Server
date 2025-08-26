import torch

from torch.cuda import is_available as cuda_is_available, is_bf16_supported

from esm.tokenization import EsmSequenceTokenizer

from esmc_function_classifier.model import EsmcGoTermClassifier

from networkx import DiGraph


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
        quantize: bool,
        quant_group_size: int,
        device: str,
    ):
        """
        Args:
            name: HuggingFace model identifier for the ESMC model.
            graph: A NetworkX DiGraph representing the Gene Ontology DAG.
            context_length: Maximum length of the input sequence.
            quantize: Whether to quantize the model weights.
            quant_group_size: Group size for quantization.
            device: Device to run the model on (e.g., "cuda" or "cpu").
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

        tokenizer = EsmSequenceTokenizer()

        model = EsmcGoTermClassifier.from_pretrained(name)

        dtype = (
            torch.bfloat16
            if "cuda" in device and is_bf16_supported()
            else torch.float16
        )

        model = model.to(device, dtype=dtype)

        model = torch.compile(model)

        if quantize:
            model.quantize_weights(group_size=quant_group_size)

        model.load_gene_ontology(graph)

        model.eval()

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

        probabilities = self.model.predict_terms(input_ids, top_p)

        return probabilities

    @torch.no_grad()
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

        subgraph, probabilities = self.model.predict_subgraph(input_ids, top_p)

        return subgraph, probabilities
