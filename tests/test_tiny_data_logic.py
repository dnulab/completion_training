from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_script(script_rel: str, args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    script = ROOT / script_rel
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_make_inputs_add_tiny_dataset_math_is_correct(tmp_path: Path) -> None:
    result = run_script("make_inputs_add.py", ["7", "8"], cwd=tmp_path)
    assert result.returncode == 0, result.stderr

    lines = read_lines(tmp_path / "inputs" / "add.txt")
    assert len(lines) == 8

    for line in lines:
        lhs, rhs = line.split("=")
        a_str, b_str = lhs.split("+")
        a = int(a_str)
        b = int(b_str)
        c = int(rhs)
        assert 0 <= a < 7
        assert 0 <= b < 7
        assert 0 <= c < 7
        assert c == (a + b) % 7


def test_make_inputs_capital_tiny_dataset_is_uppercase_transform(tmp_path: Path) -> None:
    result = run_script("make_inputs_capital.py", ["2", "3", "6"], cwd=tmp_path)
    assert result.returncode == 0, result.stderr

    lines = read_lines(tmp_path / "inputs" / "capital.txt")
    assert len(lines) == 6

    for line in lines:
        lhs, rhs = line.split("=")
        assert 1 <= len(lhs) <= 2
        assert rhs == lhs.upper()


def test_make_inputs_rev_tiny_dataset_is_reverse_transform(tmp_path: Path) -> None:
    result = run_script("make_inputs_rev.py", ["2", "3", "6"], cwd=tmp_path)
    assert result.returncode == 0, result.stderr

    lines = read_lines(tmp_path / "inputs" / "reverse.txt")
    assert len(lines) == 6

    for line in lines:
        lhs, rhs = line.split("=")
        assert len(lhs) % 2 == 0
        assert rhs == lhs[::-1]


def test_make_inputs_comp_data_is_deterministic_for_same_args(tmp_path: Path) -> None:
    args = ["2", "ab", "10"]
    output_path = tmp_path / "inputs" / "comp_data.txt"

    first = run_script("make_inputs_comp_data.py", args, cwd=tmp_path)
    assert first.returncode == 0, first.stderr
    first_content = output_path.read_text(encoding="utf-8")

    second = run_script("make_inputs_comp_data.py", args, cwd=tmp_path)
    assert second.returncode == 0, second.stderr
    second_content = output_path.read_text(encoding="utf-8")

    assert first_content == second_content

    # Within a single run, one input should map to one output consistently.
    mapping: dict[str, str] = {}
    for line in read_lines(output_path):
        lhs, rhs = line.split("=")
        if lhs in mapping:
            assert mapping[lhs] == rhs
        else:
            mapping[lhs] = rhs


def test_prepare_1char_writes_expected_outputs_and_meta(tmp_path: Path) -> None:
    input_file = tmp_path / "tiny_1char.txt"
    input_file.write_text("a=A\nb=B\ncc=CC\nab=AB\nba=BA\nac=AC\n", encoding="utf-8")

    out_dir = tmp_path / "data_1char"
    result = run_script("1-Char/prepare_1char.py", [str(input_file), "--out-dir", str(out_dir)], cwd=tmp_path)
    assert result.returncode == 0, result.stderr

    assert (out_dir / "train.bin").exists()
    assert (out_dir / "val.bin").exists()
    assert (out_dir / "meta.pkl").exists()

    with (out_dir / "meta.pkl").open("rb") as fh:
        meta = pickle.load(fh)

    assert "stoi" in meta
    assert "itos" in meta
    assert "vocab_size" in meta
    assert "_" in meta["stoi"]
    assert "=" in meta["stoi"]
    assert "\n" in meta["stoi"]


def test_prepare_2char_writes_expected_outputs_and_meta(tmp_path: Path) -> None:
    input_file = tmp_path / "tiny_2char.txt"
    input_file.write_text("a=Z\nbb=YY\nccc=WWW\n", encoding="utf-8")

    out_dir = tmp_path / "data_2char"
    result = run_script("2-Char/prepare_2char.py", [str(input_file), "--out-dir", str(out_dir)], cwd=tmp_path)
    assert result.returncode == 0, result.stderr

    assert (out_dir / "train.bin").exists()
    assert (out_dir / "val.bin").exists()
    assert (out_dir / "meta.pkl").exists()

    with (out_dir / "meta.pkl").open("rb") as fh:
        meta = pickle.load(fh)

    stoi = meta["stoi"]
    assert "=" in stoi
    assert "\n" in stoi
    assert "_" in stoi
    # Odd-length tokens should produce padded two-char tokens in the vocabulary.
    assert "a_" in stoi
    assert "Z_" in stoi
    assert "cc" in stoi
    assert "c_" in stoi
    assert "WW" in stoi
    assert "W_" in stoi
