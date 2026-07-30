from __future__ import annotations

from pathlib import Path

import torch

from completion_core.inference import complete_sequence
from completion_core.modeling import load_model_and_config
from completion_core.vocabulary import Vocabulary


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path("out_1char")
    data_dir = Path("1-Char/data")
    weights_path = out_dir / "completion_model.pth"

    if not weights_path.exists():
        raise FileNotFoundError(f"Could not find model weights at {weights_path}")

    vocab = Vocabulary.from_pickle(data_dir / "meta.pkl")
    model, model_cfg = load_model_and_config(
        weights_path=weights_path,
        vocab=vocab,
        device=device,
    )

    print("Model weights successfully loaded! Running systematic evaluation...\n")

    correct_predictions = 0
    total_predictions = 0

    print("Evaluating all pairs from 0+0 to 299+299...")

    for i in range(0, 300):
        for j in range(0, 300):
            lhs = f"{i}+{j}"
            ground_truth = str((i + j) % 300)

            prediction = complete_sequence(
                lhs=lhs,
                model=model,
                vocab=vocab,
                max_new_tokens=len(ground_truth) + 1,
                max_supported_len=model_cfg.seq_len,
                device=device,
            )

            if prediction == ground_truth:
                correct_predictions += 1

            total_predictions += 1

    print("\n" + "=" * 40)
    print("EXHAUSTIVE ACCURACY REPORT (generate_all.py)")
    print(f"Total Combinations Evaluated : {total_predictions}")
    print(f"Total Correct Math Matches   : {correct_predictions}")
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"Overall Model Accuracy       : {accuracy:.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    main()