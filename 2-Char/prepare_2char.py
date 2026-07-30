"""
Prepare a 2-character level dataset for train_completions.py.

Reads a plain-text file where each line is one sequence of format {input}={output},
splits it 90/10 into train and validation sets, builds a custom 2-character 
vocabulary (treating '=' and '\n' as separate tokens), and writes 
train.bin, val.bin, and meta.pkl to an output directory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from completion_core.prep import (
    build_token_vocab,
    encode_token_lines,
    load_lines,
    save_outputs,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PrepareConfig:
    input_file: Path = Path("inputs/comp_data.txt")
    out_dir: Path = Path("data")
    train_split: float = 0.9
    pad_token: str = "_"        # Used to pad odd-length inputs/outputs


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def tokenize_line(line: str, pad_token: str) -> list[str]:
    """
    Splits a line into custom 2-character tokens, keeping '=' and '\\n' separate.
    Pads odd-lengthed input or output parts with pad_token.
    """
    if '=' not in line:
        raise ValueError(f"Line missing '=' separator: {line}")
        
    inp, out = line.split('=', 1)
    tokens = []
    
    # Process Input Side
    if len(inp) % 2 != 0:
        inp += pad_token
    for i in range(0, len(inp), 2):
        tokens.append(inp[i:i+2])
        
    # Separator
    tokens.append('=')
    
    # Process Output Side
    if len(out) % 2 != 0:
        out += pad_token
    for i in range(0, len(out), 2):
        tokens.append(out[i:i+2])
        
    # End of line token
    tokens.append('\n')
    
    return tokens


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> PrepareConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_file",
        nargs="?",
        type=Path,
        default=Path("inputs/comp_data.txt"),
        help="Path to the input text file (default: inputs/comp_data.txt)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data"),
        help="Directory to write train.bin, val.bin, meta.pkl (default: data/)",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.9,
        help="Fraction of lines used for training (default: 0.9)",
    )
    args = parser.parse_args()

    cfg = PrepareConfig()
    cfg.input_file = args.input_file
    cfg.out_dir = args.out_dir
    cfg.train_split = args.train_split
    return cfg


def main() -> None:
    cfg = parse_args()

    lines = load_lines(cfg.input_file, strip=True, drop_empty=True)
    print(f"Lines loaded    : {len(lines)}")
    
    # Tokenize every line individual first
    tokenized_lines = [tokenize_line(line, cfg.pad_token) for line in lines]

    # Split dataset based on lines
    split = int(len(tokenized_lines) * cfg.train_split)
    train_lines = tokenized_lines[:split]
    val_lines = tokenized_lines[split:]

    # Build vocabulary using all tokens
    stoi, itos = build_token_vocab(tokenized_lines, cfg.pad_token)
    print(f"Vocabulary size : {len(stoi)}")

    # --- ADD THIS TO DUMP THE VOCAB ---
    import pprint
    print("\n--- Vocabulary Mapping (stoi) ---")
    pprint.pprint(stoi)
    print("----------------------------------\n")
    # ----------------------------------
    
    # Flatten and convert tokens to integer IDs
    train_ids = encode_token_lines(train_lines, stoi)
    val_ids = encode_token_lines(val_lines, stoi)
    print(f"Train tokens    : {len(train_ids)}")
    print(f"Val tokens      : {len(val_ids)}")

    save_outputs(cfg.out_dir, train_ids, val_ids, stoi, itos)
    print(f"\nFiles written to '{cfg.out_dir}/'")


if __name__ == "__main__":
    main()