from pydantic import BaseModel, Field

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

    quant_group_size: int = Field(
        description="The group size used for quantization.",
    )

    max_concurrency: int = Field(
        description="The maximum number of concurrent generations.",
    )


class PredictTermsRequest(BaseModel):
    sequences: list[str] = Field(
        min_length=1, description="A list of protein amino acid sequences."
    )

    top_p: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="The minimum probability threshold for GO term predictions.",
    )


class PredictMfTermsRequest(PredictTermsRequest):
    pass


class PredictBpTermsRequest(PredictTermsRequest):
    pass


class PredictCcTermsRequest(PredictTermsRequest):
    pass


class PredictAllTermsRequest(PredictTermsRequest):
    pass


class PredictTermsResponse(BaseModel):
    terms: list[list[dict[str, float]]] = Field(
        description="List of GO terms and their probabilities."
    )


class PredictMfTermsResponse(PredictTermsResponse):
    pass


class PredictBpTermsResponse(PredictTermsResponse):
    pass


class PredictCcTermsResponse(PredictTermsResponse):
    pass


class PredictAllTermsResponse(BaseModel):
    mf_terms: list[list[dict[str, float]]] = Field(
        description="List of MF GO terms and their probabilities."
    )

    bp_terms: list[list[dict[str, float]]] = Field(
        description="List of BP GO terms and their probabilities."
    )

    cc_terms: list[list[dict[str, float]]] = Field(
        description="List of CC GO terms and their probabilities."
    )


class PredictSubgraphsRequest(BaseModel):
    sequences: list[str] = Field(
        min_length=1, description="A list of protein amino acid sequences."
    )

    top_p: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="The minimum probability threshold for GO term predictions.",
    )


class PredictMfSubgraphsRequest(PredictSubgraphsRequest):
    pass


class PredictBpSubgraphsRequest(PredictSubgraphsRequest):
    pass


class PredictCcSubgraphsRequest(PredictSubgraphsRequest):
    pass


class PredictAllSubgraphsRequest(PredictSubgraphsRequest):
    pass


class PredictSubgraphsResponse(BaseModel):
    subgraphs: list[dict] = Field(
        description="A subgraph of the gene ontology in node-link format."
    )

    terms: list[dict[str, float]] = Field(
        description="List of predicted GO terms and their probabilities."
    )


class PredictMfSubgraphsResponse(PredictSubgraphsResponse):
    pass


class PredictBpSubgraphsResponse(PredictSubgraphsResponse):
    pass


class PredictCcSubgraphsResponse(PredictSubgraphsResponse):
    pass


class PredictAllSubgraphsResponse(BaseModel):
    mf_subgraphs: list[dict] = Field(
        description="A list of subgraphs of the MF aspect of the gene ontology in node-link format."
    )

    bp_subgraphs: list[dict] = Field(
        description="A list of subgraphs of the BP aspect of the gene ontology in node-link format."
    )

    cc_subgraphs: list[dict] = Field(
        description="A list of subgraphs of the CC aspect of the gene ontology in node-link format."
    )

    mf_terms: list[dict[str, float]] = Field(
        description="List of MF GO terms and their probabilities."
    )

    bp_terms: list[dict[str, float]] = Field(
        description="List of BP GO terms and their probabilities."
    )

    cc_terms: list[dict[str, float]] = Field(
        description="List of CC GO terms and their probabilities."
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
        quant_group_size=model.quant_group_size,
        max_concurrency=model.max_concurrency,
    )


@router.post("/gene-ontology/aspects/mf/terms", response_model=PredictMfTermsResponse)
async def predict_mf_terms(request: Request, input: PredictMfTermsRequest):
    """Return all the GO term probabilities for a protein sequence."""

    terms = request.app.state.model.predict_mf_terms(input.sequences, input.top_p)

    return PredictMfTermsResponse(terms=terms)


@router.post("/gene-ontology/aspects/bp/terms", response_model=PredictBpTermsResponse)
async def predict_bp_terms(request: Request, input: PredictBpTermsRequest):
    """Return all the GO term probabilities for a protein sequence."""

    terms = request.app.state.model.predict_bp_terms(input.sequences, input.top_p)

    return PredictBpTermsResponse(terms=terms)


@router.post("/gene-ontology/aspects/cc/terms", response_model=PredictCcTermsResponse)
async def predict_cc_terms(request: Request, input: PredictCcTermsRequest):
    """Return all the GO term probabilities for a protein sequence."""

    terms = request.app.state.model.predict_cc_terms(input.sequences, input.top_p)

    return PredictCcTermsResponse(terms=terms)


@router.post("/gene-ontology/aspects/all/terms", response_model=PredictAllTermsResponse)
async def predict_all_terms(request: Request, input: PredictAllTermsRequest):
    """Return all the GO term probabilities for a protein sequence."""

    mf_terms, bp_terms, cc_terms = request.app.state.model.predict_all_terms(
        input.sequences, input.top_p
    )

    return PredictAllTermsResponse(
        mf_terms=mf_terms, bp_terms=bp_terms, cc_terms=cc_terms
    )


@router.post(
    "/gene-ontology/aspects/mf/subgraphs", response_model=PredictMfSubgraphsResponse
)
async def predict_mf_subgraphs(request: Request, input: PredictMfSubgraphsRequest):
    """Return all the GO MF subgraphs for a protein sequence."""

    subgraphs, terms = request.app.state.model.predict_mf_subgraphs(
        input.sequences, input.top_p
    )

    subgraphs = [nx.node_link_data(subgraph, edges="edges") for subgraph in subgraphs]

    return PredictMfSubgraphsResponse(subgraphs=subgraphs, terms=terms)


@router.post(
    "/gene-ontology/aspects/bp/subgraphs", response_model=PredictBpSubgraphsResponse
)
async def predict_bp_subgraphs(request: Request, input: PredictBpSubgraphsRequest):
    """Return all the GO BP subgraphs for a protein sequence."""

    subgraphs, terms = request.app.state.model.predict_bp_subgraphs(
        input.sequences, input.top_p
    )

    subgraphs = [nx.node_link_data(subgraph, edges="edges") for subgraph in subgraphs]

    return PredictBpSubgraphsResponse(subgraphs=subgraphs, terms=terms)


@router.post(
    "/gene-ontology/aspects/cc/subgraphs", response_model=PredictCcSubgraphsResponse
)
async def predict_cc_subgraphs(request: Request, input: PredictCcSubgraphsRequest):
    """Return all the GO CC subgraphs for a protein sequence."""

    subgraphs, terms = request.app.state.model.predict_cc_subgraphs(
        input.sequences, input.top_p
    )

    subgraphs = [nx.node_link_data(subgraph, edges="edges") for subgraph in subgraphs]

    return PredictCcSubgraphsResponse(subgraphs=subgraphs, terms=terms)


@router.post(
    "/gene-ontology/aspects/all/subgraphs", response_model=PredictAllSubgraphsResponse
)
async def predict_all_subgraphs(request: Request, input: PredictAllSubgraphsRequest):
    """Return all the GO subgraphs for a protein sequence."""

    mf_results, bp_results, cc_results = request.app.state.model.predict_all_subgraphs(
        input.sequences, input.top_p
    )

    mf_subgraphs, mf_terms = mf_results
    bp_subgraphs, bp_terms = bp_results
    cc_subgraphs, cc_terms = cc_results

    mf_subgraphs = [
        nx.node_link_data(subgraph, edges="edges") for subgraph in mf_subgraphs
    ]

    bp_subgraphs = [
        nx.node_link_data(subgraph, edges="edges") for subgraph in bp_subgraphs
    ]

    cc_subgraphs = [
        nx.node_link_data(subgraph, edges="edges") for subgraph in cc_subgraphs
    ]

    return PredictAllSubgraphsResponse(
        mf_subgraphs=mf_subgraphs,
        bp_subgraphs=bp_subgraphs,
        cc_subgraphs=cc_subgraphs,
        mf_terms=mf_terms,
        bp_terms=bp_terms,
        cc_terms=cc_terms,
    )
