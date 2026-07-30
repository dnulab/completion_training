#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random

from completion_core.dataset_generators import resolve_charset, write_lines


def generate_all_unique_strings(n: int, charset: str) -> list[str]:
    pool: list[str] = []

    def build(current_str: str) -> None:
        if len(current_str) > 0:
            pool.append(current_str)
        if len(current_str) == n:
            return
        for char in charset:
            build(current_str + char)

    build("")
    return pool


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic 1-to-1 mapping dataset."
    )
    parser.add_argument("n", type=int, help="Max string length")
    parser.add_argument(
        "raw_charset",
        type=str,
        help="An integer size for ascii_lowercase slice, or a literal string of characters to use",
    )
    parser.add_argument("num_lines", type=int, help="Number of dataset lines to generate")

    args = parser.parse_args()

    charset = resolve_charset(args.raw_charset)
    all_possible_inputs = generate_all_unique_strings(args.n, charset)

    random.seed(42)
    all_possible_outputs = all_possible_inputs.copy()
    random.shuffle(all_possible_outputs)
    mapping_dict = dict(zip(all_possible_inputs, all_possible_outputs))

    def build_line() -> str:
        a = random.choice(all_possible_inputs)
        return f"{a}={mapping_dict[a]}\n"

    write_lines("input.txt", args.num_lines, build_line)
    print(f"Successfully generated {args.num_lines} lines in input.txt")


if __name__ == "__main__":
    main()