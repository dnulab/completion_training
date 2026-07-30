#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys

from completion_core.cli import print_default_args_message
from completion_core.dataset_generators import (
    bounded_charset,
    print_generation_header,
    random_string,
    write_lines,
)


def generate_capital_dataset(n: int, charset_size: int, num_lines: int) -> None:
    available_chars = bounded_charset(charset_size)
    output_filename = "inputcapital.txt"

    print_generation_header(n, available_chars, num_lines, output_filename)

    def build_line() -> str:
        line_len = random.randint(1, n)
        input_str = random_string(available_chars, line_len)
        return f"{input_str}={input_str.upper()}\n"

    write_lines(output_filename, num_lines, build_line)
    print("Dataset successfully created")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a capitalization dataset for sequence completion models."
    )
    parser.add_argument("n", type=int, nargs="?", default=3, help="Max string length (default: 3)")
    parser.add_argument(
        "charset_size",
        type=int,
        nargs="?",
        default=26,
        help="Size of the alphabet charset to use (default: 26)",
    )
    parser.add_argument(
        "num_lines",
        type=int,
        nargs="?",
        default=1000,
        help="Number of dataset lines to generate (default: 1000)",
    )

    args = parser.parse_args()
    if len(sys.argv) < 2:
        print_default_args_message(args.n, args.charset_size, args.num_lines)

    generate_capital_dataset(args.n, args.charset_size, args.num_lines)


if __name__ == "__main__":
    main()