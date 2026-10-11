"""One-time deploy-time setup for Flower Image Search.

Pre-builds the OpenCLIP search index so the first real visitor never has to
wait through a dataset download + embedding pass — the app just loads the
already-built Chroma collection at request time.

Run once after cloning / before first deploy:
    python scripts/build_flower_index.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.flower_index import build_index  # noqa: E402


def main() -> None:
    count = build_index()
    print(f"Flower search index ready: {count} photos indexed in data/flower_db.")


if __name__ == "__main__":
    main()
