from os import environ

import torch

import obonet

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from middleware import ExceptionHandler, TokenAuthentication, ResponseTime

from routers import go_classifier, health

from model import GoTermClassifier

import uvicorn


api_token = environ.get("API_TOKEN", "")
model_name = environ.get("MODEL_NAME", "andrewdalpino/ESMC-300M-Protein-Function")
go_db_path = environ.get("GO_DB_PATH", "./dataset/go-basic.obo")
context_length = int(environ.get("CONTEXT_LENGTH", 2048))
device = environ.get("DEVICE", "cpu")
dtype = environ.get("DTYPE", "float32")

app = FastAPI(
    title="ESMC GO Inference Server",
    description="Inference server for protein gene ontology (GO) classification using the EMC Cambrian family of models.",
    version="0.0.3",
)

graph = obonet.read_obo(go_db_path)

match dtype:
    case "float32":
        dtype = torch.float32
    case "bfloat16":
        dtype = torch.bfloat16
    case "float16":
        dtype = torch.float16
    case _:
        raise ValueError(
            f"Unsupported dtype: {dtype}. Supported dtypes are float32, bfloat16, and float16."
        )

model = GoTermClassifier(
    model_name=model_name,
    graph=graph,
    context_length=context_length,
    device=device,
    dtype=dtype,
)

app.state.model = model

app.add_middleware(ExceptionHandler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if api_token:
    app.add_middleware(TokenAuthentication, api_token=api_token)

app.include_router(go_classifier.router)
app.include_router(health.router)

app.add_middleware(ResponseTime)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
