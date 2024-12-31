import argparse
import json

from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def evaluate():
    texts, entities = [], []
    with open(args.eval_file, "r") as f:
        for line in f:
            line = json.loads(line)
            texts.append(line["text"].strip())
            entities.append(set([(e["label"], e["start_offset"], e["end_offset"]) for e in line["entities"]]))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    predictions = model.predict(
        tokenizer,
        texts,
        batch_size=args.batch_size,
        max_length=args.max_seq_len,
        show_progress_bar=True,
        device=args.device,
    )
    res = []
    for pred in predictions:
        res.append(set([(e.type, e.start, e.end) for e in pred]))

    X, Y, Z = 1e-10, 1e-10, 1e-10
    for R, T in tqdm(zip(entities, res), ncols=100):
        X += len(R & T)
        Y += len(R)
        Z += len(T)

    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z

    print("{:<15} {:<15}".format("Metric", "Score"))
    print("-" * 30)
    print("{:<15} {:<15.4f}".format("F1 Score", f1))
    print("{:<15} {:<15.4f}".format("Precision", precision))
    print("{:<15} {:<15.4f}".format("Recall", recall))


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
        default="cpu",
        help="Select which device to run model, defaults to gpu."
    )

    args = parser.parse_args()

    evaluate()
