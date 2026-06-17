#!/usr/bin/env python3
from __future__ import annotations

import sys

from run_megabyte_window_roundtrip import main


def _translate_legacy_args(argv: list[str]) -> list[str]:
    translated: list[str] = []
    skip_next = False
    for index, arg in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if arg == "--windows":
            translated.append("--random-windows")
            continue
        if arg == "--tokens":
            translated.append("--tokens-per-window")
            continue
        if arg == "--skip-decode":
            print(
                "[compat] --skip-decode is no longer supported by this wrapper; "
                "use run_megabyte_window_compress.py for compression-only runs.",
                file=sys.stderr,
            )
            continue
        translated.append(arg)
    return translated


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *_translate_legacy_args(sys.argv[1:])]
    main()
