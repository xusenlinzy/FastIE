import gc
import os
from contextlib import asynccontextmanager
from typing import Optional

import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModel,
    AutoTokenizer,
)
from transformers.utils import is_torch_cuda_available
from transformers.utils.import_utils import _is_package_available
from typing_extensions import Annotated

from .protocol import IECreateParams, get_response_model
from ..hparams.parser import get_infer_args

if _is_package_available("fastapi"):
    from fastapi import Depends, FastAPI, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer


if _is_package_available("uvicorn"):
    import uvicorn


def torch_gc() -> None:
    r"""
    Collects GPU or NPU memory.
    """
    gc.collect()
    if is_torch_cuda_available():
        torch.cuda.empty_cache()


@asynccontextmanager
async def lifespan(app: "FastAPI"):  # collects GPU memory
    yield
    torch_gc()


def create_app(model: "PreTrainedModel", tokenizer: "PreTrainedTokenizer") -> "FastAPI":
    root_path = os.environ.get("FASTAPI_ROOT_PATH", "")
    app = FastAPI(lifespan=lifespan, root_path=root_path)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    api_key = os.environ.get("API_KEY")
    security = HTTPBearer(auto_error=False)

    async def verify_api_key(auth: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)]):
        if api_key and (auth is None or auth.credentials != api_key):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key.")

    architecture = model.config.architectures[0].lower()
    response_model = get_response_model(architecture)

    @app.post(
        "/v1/ie",
        response_model=response_model,
        status_code=status.HTTP_200_OK,
        dependencies=[Depends(verify_api_key)],
    )
    async def create_ie(request: IECreateParams):
        if request.ie_schema and "uie" in architecture:
            model.set_schema(request.ie_schema)

        labels = model.predict(
            tokenizer,
            request.texts,
            batch_size=request.batch_size,
            max_length=request.max_length,
            language=request.language,
        )
        return response_model(model=model.config.architectures[0], labels=labels)

    return app


def run_api() -> None:
    model_args, infer_args = get_infer_args()
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    ).to(model_args.device)

    app = create_app(model, tokenizer)
    print("Visit http://localhost:{}/docs for API document.".format(infer_args.port))
    uvicorn.run(app, host=infer_args.host, port=infer_args.port)
