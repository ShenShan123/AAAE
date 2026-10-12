#!/usr/bin/env python3
import argparse
import _bootstrap
from aaae.cli import run_module

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rerun selected designs using the fixed accuracy protocol.")
    parser.add_argument("--family", choices=["search", "transformer", "cnn"], required=True)
    options, remaining = parser.parse_known_args()
    if remaining[:1] == ["--"]:
        remaining = remaining[1:]
    module = {"search": "search_candidates", "transformer": "transformer", "cnn": "cnn"}[options.family]
    raise SystemExit(run_module(f"aaae.evaluation.verification.{module}", remaining))
