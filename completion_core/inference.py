from __future__ import annotations

from pathlib import Path

import torch

from completion_core.modeling import CompletionTransformer
from completion_core.vocabulary import Vocabulary


@torch.no_grad()
def complete_sequence(
    lhs: str,
    model: CompletionTransformer,
    vocab: Vocabulary,
    max_new_tokens: int,
    max_supported_len: int,
    device: str,
) -> str:
    lhs_tokens = vocab.tokenize_string(lhs)
    tokens = [vocab.stoi[tok] for tok in lhs_tokens] + [vocab.equal_id]

    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    start_len = x.size(1)

    while (x.size(1) - start_len) < max_new_tokens and x.size(1) < max_supported_len:
        logits = model(x)
        next_id = torch.argmax(logits[0, -1, :]).item()
        x = torch.cat([x, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == vocab.eot_id:
            break

    gen_tokens = [vocab.itos[t.item()] for t in x[0]][start_len:]
    return "".join(gen_tokens).replace("\n", "").replace("_", "")


def evaluate_file(
    eval_file: Path,
    model: CompletionTransformer,
    vocab: Vocabulary,
    max_new_tokens: int,
    max_supported_len: int,
    device: str,
) -> tuple[int, int]:
    if not eval_file.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_file}")

    print(f"Evaluating: {eval_file}\n")

    correct = 0
    total = 0

    with eval_file.open(encoding="utf-8") as fh:
        for line_num, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if "=" not in line:
                print(f"  [SKIP] Line {line_num}: missing '=' - '{raw_line.strip()}'")
                continue

            lhs, ground_truth = line.split("=", maxsplit=1)
            lhs_clean = lhs.replace("_", "")
            gt_clean = ground_truth.replace("_", "")

            gt_tokens_len = len(vocab.tokenize_string(gt_clean)) + 1
            dynamic_new_tokens = max_new_tokens if not vocab.is_2char else gt_tokens_len

            prediction = complete_sequence(
                lhs=lhs_clean,
                model=model,
                vocab=vocab,
                max_new_tokens=dynamic_new_tokens,
                max_supported_len=max_supported_len,
                device=device,
            )

            if prediction == gt_clean:
                correct += 1
            else:
                print(
                    f"  [MISS] Prompt: {lhs_clean + '=':<6}  "
                    f"Expected: {gt_clean:<5}  Got: {prediction:<5}"
                )
            total += 1

    return correct, total
