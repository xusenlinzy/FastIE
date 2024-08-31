import argparse
import json

from sklearn.metrics import classification_report
from transformers import AutoModel, AutoTokenizer


def evaluate():
    texts_a, texts_b, labels = [], [], []
    with open(args.eval_file, "r") as f:
        for line in f:
            line = json.loads(line)
            if "sentence1" in line and "sentence2" in line:
                texts_a.append(line["sentence1"].strip())
                texts_b.append(line["sentence2"].strip())
            else:
                texts_a.append(line["text"].strip())
            labels.append(line["label"].strip())

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModel.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, device_map=args.device
    )
    predictions = model.predict(
        tokenizer,
        texts_a,
        text_b=texts_b if texts_b else None,
        batch_size=args.batch_size,
        max_length=args.max_seq_len,
    )

    print(classification_report(labels, predictions))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model_name_or_path",
        type=str,
        required=True,
        help="The path prefix of inference model to be used.",
    )
    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
    )
    parser.add_argument(
        "--max_seq_len",
        default=512,
        type=int,
        help="The maximum input sequence length. Sequences longer than this will be split automatically.",
    )
    parser.add_argument(
        "-D",
        "--device",
        choices=['cpu', 'cuda'],
        default="cpu",
        help="Select which device to run model, defaults to gpu."
    )

    args = parser.parse_args()

    evaluate()
