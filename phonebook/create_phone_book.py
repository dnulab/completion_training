#!/usr/bin/env python3
"""Create a randomized synthetic phone book."""

import argparse
import random
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FIRST_NAMES = SCRIPT_DIR / "names" / "first_names.txt"
DEFAULT_LAST_NAMES = SCRIPT_DIR / "names" / "last_names.txt"
DEFAULT_OUTPUT = Path("inputs") / "phone_book.txt"
DEFAULT_SEED = 104729


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a text file containing random, unique phone-book entries."
    )
    parser.add_argument("count", type=int, help="number of entries to create")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "-s",
        "--separator",
        default="=",
        help="name/number separator (default: =)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"random-number seed (default: {DEFAULT_SEED})",
    )
    return parser.parse_args()


def read_names(path: Path) -> list[str]:
    try:
        names = [line.strip() for line in path.read_text(encoding="ascii").splitlines()]
    except OSError as exc:
        raise SystemExit(f"Could not read {path}: {exc}") from exc

    names = [name for name in names if name]
    if not names:
        raise SystemExit(f"Name file is empty: {path}")
    return names


def format_phone_number(value: int) -> str:
    # Both the area-code and exchange prefixes begin with 2–9.
    area = 200 + value % 800
    value //= 800
    exchange = 200 + value % 800
    subscriber = (value // 800) % 10_000
    return f"{area:03d}-{exchange:03d}-{subscriber:04d}"


def main() -> None:
    args = parse_args()
    if args.count < 0:
        raise SystemExit("count must be zero or greater")
    if "\n" in args.separator or "\r" in args.separator:
        raise SystemExit("separator cannot contain a newline")

    first_names = read_names(DEFAULT_FIRST_NAMES)
    last_names = read_names(DEFAULT_LAST_NAMES)
    available_names = len(first_names) * len(last_names)
    if args.count > available_names:
        raise SystemExit(
            f"count cannot exceed {available_names:,}, the number of unique name pairs"
        )

    rng = random.Random(args.seed)
    name_indexes = rng.sample(range(available_names), args.count)

    # There are 6.4 billion numbers in this constrained 10-digit format.
    phone_indexes = rng.sample(range(800 * 800 * 10_000), args.count)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="ascii", newline="\n") as output:
        for name_index, phone_index in zip(name_indexes, phone_indexes):
            first = first_names[name_index // len(last_names)]
            last = last_names[name_index % len(last_names)]
            phone = format_phone_number(phone_index)
            output.write(f"{first} {last}{args.separator}{phone}\n")

    print(f"Created {args.count:,} entries in {args.output}")


if __name__ == "__main__":
    main()
