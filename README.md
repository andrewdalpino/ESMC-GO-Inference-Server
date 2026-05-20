# ESMC Protein Function Inference Server

A FastAPI application that serves ESM (Evolutionary Scale Modeling) Cambrian Protein Function models.

## Environment Variables

| Name | Default | Type | Description |
| --- | --- | --- | --- |
| API_TOKEN | None | str | The API token used to authenticate requests. |
| MODEL_NAME | "andrewdalpino/ESMC-Protein-Function-V1-300M" | str | The name of the pretrained model to load from HuggingFace Hub. |
| GO_DB_PATH | /opt/dataset/go-basic.obo | str | The path to the gene ontology database file. |
| CONTEXT_LENGTH | 2048 | int | The maximum number of tokens to process at the same time. |
| QUANTIZE | false | bool | Should we quantize the weights of the model? |
| QUANT_GROUP_SIZE | 64 | int | The size of the quantization group. |
| DEVICE | "cpu" | str | The name of the device to load the model weights onto. |
| MAX_CONCURRENCY | 1 | int | The maximum number of requests to handle at the same time. |
