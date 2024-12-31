
import time
from typing import List, Union

import gradio as gr
from transformers import AutoModel, AutoTokenizer

from ..hparams.parser import get_infer_args


class PlayGround:
    def __init__(
        self,
        model_path: str,
        device: str,
        server_name="0.0.0.0",
        server_port=7860,
        title=None,
        **kwargs,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)

        self.server_name = server_name
        self.server_port = server_port

        self.title = title
        self.kwargs = kwargs

        if self.title is None:
            self.title = "Fast Information Extraction Demo"

    def extract(self, texts: Union[str, List[str]], language: str, use_cuda=False, max_length=512, ie_schema=None):
        architecture = self.model.config.architectures[0].lower()
        if ie_schema and "uie" in architecture:
            self.model.set_schema(eval(ie_schema.strip()))

        start = time.time()
        res = self.model.predict(
            self.tokenizer,
            texts,
            batch_size=32,
            max_length=max_length,
            language=language,
            device="cuda:0" if use_cuda else "cpu",
        )

        return time.time() - start, res

    def launch(self) -> None:
        self.demo.launch(server_name=self.server_name, server_port=self.server_port, **self.kwargs)

    @property
    def demo(self):
        return gr.Interface(
            self.extract,
            [
                gr.Textbox(
                    placeholder="Enter sentence here...",
                    lines=5,
                    label="文本"
                ),
                gr.Dropdown(
                    choices=["zh", "en"],
                    value="zh",
                    label="语言"
                ),
                gr.Checkbox(label="使用GPU"),
                gr.Slider(10, 512, value=512, label="最大长度"),
                gr.Text(label="UIE模型SCHEMA", placeholder='["时间", "地点"]')
            ],
            [gr.Label(label="推理时间（s）"), gr.Json(label="抽取结果")],
            title=self.title,
        )


def run_web_demo() -> None:
    model_args, infer_args = get_infer_args()
    demo = PlayGround(
        model_args.model_name_or_path,
        model_args.device,
        server_name=infer_args.host,
        server_port=infer_args.port,
    )
    demo.launch()
