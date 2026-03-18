from __future__ import annotations

import argparse
from pathlib import Path

from item6_analysis import run_item6
from item7_comparison import run_item7
from item8_model_selection import run_item8
from item9_causal_adjustment import run_item9


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to input CSV dataset")
    parser.add_argument("--output", default="output", help="Directory to store outputs")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_item6(args.data, output_dir)
    run_item7(args.data, output_dir)
    run_item8(args.data, output_dir)
    run_item9(args.data, output_dir)

    print(f"All analyses completed. Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
