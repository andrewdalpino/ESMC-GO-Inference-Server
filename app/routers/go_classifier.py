from pydantic import BaseModel, Field
from typing import Any

from fastapi import APIRouter, Request

import networkx as nx


class ModelInfoResponse(BaseModel):
    name: str = Field(
        description="The name of the ESM3 model variant.",
    )

    num_parameters: int = Field(
        description="The number of parameters in the model.",
    )

    context_length: int = Field(
        description="The context length of the input sequences.",
    )

    device: str = Field(
        description="The device the model is running on.",
    )

    quantize: bool = Field(
        description="Whether the model weights are quantized to Int8.",
    )

    max_concurrency: int = Field(
        description="The maximum number of concurrent generations.",
    )


class PredictTermsRequest(BaseModel):
    sequence: str = Field(min_length=1, description="A protein amino acid sequence.")

    top_p: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="The minimum probability threshold for GO term predictions.",
    )


class PredictTermsResponse(BaseModel):
    probabilities: dict[str, float] = Field(
        description="List of GO terms and their probabilities."
    )


class PredictSubgraphRequest(BaseModel):
    sequence: str = Field(min_length=1, description="A protein amino acid sequence.")

    top_p: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="The minimum probability threshold for GO term predictions.",
    )


class PredictSubgraphResponse(BaseModel):
    subgraph: dict = Field(
        description="A subgraph of the gene ontology in node-link format."
    )

    probabilities: dict[str, float] = Field(
        description="List of predicted GO terms and their probabilities."
    )


router = APIRouter(prefix="/model")


@router.get("/", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    model = request.app.state.model

    return ModelInfoResponse(
        name=model.name,
        num_parameters=model.num_parameters,
        context_length=model.context_length,
        device=model.device,
        quantize=model.quantize,
        max_concurrency=model.max_concurrency,
    )


@router.post("/gene-ontology/terms")
async def predict_terms(request: Request, input: PredictTermsRequest):
    """Return the GO term probabilities for a protein sequence."""

    go_term_probabilities = request.app.state.model.predict_terms(
        input.sequence, input.top_p
    )

    return PredictTermsResponse(probabilities=go_term_probabilities)


@router.post("/gene-ontology/subgraph")
async def predict_subgraph(request: Request, input: PredictSubgraphRequest):
    """Return the GO subgraph for a protein sequence."""

    subgraph, go_term_probabilities = request.app.state.model.predict_subgraph(
        input.sequence, input.top_p
    )

    subgraph = nx.node_link_data(subgraph, edges="edges")

    return PredictSubgraphResponse(
        subgraph=subgraph,
        probabilities=go_term_probabilities,
    )
