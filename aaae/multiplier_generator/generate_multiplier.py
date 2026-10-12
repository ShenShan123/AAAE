"""Generate an 8-bit multiplier in AAAE's fixed low-only design family."""

from __future__ import annotations

import argparse
from pathlib import Path

from aaae.paths import OUTPUT_ROOT
from .nbit_approx import MultiplierGeneratorArbitrary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compressors", nargs=9, type=int, choices=range(1, 9), required=True,
                        metavar="ID", help="compressor IDs at the nine configurable low-order positions")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT / "multiplier" / "multiplier.v")
    args = parser.parse_args()
    generator = MultiplierGeneratorArbitrary(
        n_bits=8, compressor_types=args.compressors, approx_high_part=False,
    )
    rtl = generator.generate_verilog_code()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rtl)
    print(f"Generated 8-bit multiplier: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
