from typing import Optional, Any

import litserve as ls
import torch
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer, AutoConfig

from fastie.api.protocol import get_response_model
from fastie.hparams.parser import get_infer_args

model_args, infer_args = get_infer_args()
config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
if "UIE" in config.architectures[0]:
    infer_args.max_batch_size = 1


class IECreateParams(BaseModel):
    text: str
    ie_schema: Optional[Any] = None


class FastIEAPI(ls.LitAPI):
    def setup(self, device):
        """Load the tokenizer and model, and move the model to the specified device."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
        self.model.to(device)
        self.model.eval()

        self.architecture = config.architectures[0]
        print(f"Loading Model Architecture: {self.architecture}")
        self.response_model = get_response_model(self.architecture.lower())

    def decode_request(self, request: IECreateParams, **kwargs):
        """Convert the request payload to your model input."""
        return {"text": request.text, "schema": request.ie_schema}

    @torch.inference_mode()
    def predict(self, x, **kwargs):
        """Run the model on the input and return or yield the output."""
        if isinstance(x, list):
            assert infer_args.max_batch_size > 1, "Unexpected batch request recieved!"
            texts = [d["text"] for d in x]
            schema = None
        else:
            texts, schema = x["text"], x["schema"]

        kwargs = {"texts": texts}
        if "UIE" in self.architecture:
            kwargs["schema"] = schema

        return self.model.predict(self.tokenizer, **kwargs)

    def encode_response(self, output, **kwargs):
        """Convert the model output to a response payload."""
        return self.response_model(model=self.architecture, labels=output)


if __name__ == "__main__":
    api = FastIEAPI()
    server = ls.LitServer(
        api,
        accelerator=infer_args.accelerator,
        devices=infer_args.devices,
        max_batch_size=infer_args.max_batch_size,
        workers_per_device=infer_args.workers_per_device,
        api_path="/v1/ie",
        timeout=infer_args.timeout,
        batch_timeout=infer_args.batch_timeout,
    )
    server.run(
        port=infer_args.port,
        num_api_servers=infer_args.num_api_servers,
        generate_client_file=False,
    )
