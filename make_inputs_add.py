#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random

from completion_core.dataset_generators import default_inputs_path, write_lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a modular addition dataset."
    )
    parser.add_argument(
        "V",
        type=int,
        help="The modulo base value (upper bound for random integers)",
    )
    parser.add_argument(
        "N",
        type=int,
        help="Number of addition lines to generate",
    )
    args = parser.parse_args()

    def build_line() -> str:
        a = random.randint(0, args.V - 1)
        b = random.randint(0, args.V - 1)
        return f"{a}+{b}={(a + b) % args.V}\n"

    write_lines(default_inputs_path("add.txt"), args.N, build_line)


if __name__ == "__main__":
    main()