from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Iterator


def get_example_io_docs_path() -> Path:
    """Return the packaged example I/O markdown directory."""
    return Path(
        resources.files("diaad.examples").joinpath(
            "assets",
            "rendered_docs",
            "example_io",
        )
    )


def iter_example_io_markdown_files() -> Iterator[Path]:
    """Yield packaged example I/O markdown files in display order."""
    docs_path = get_example_io_docs_path()
    if not docs_path.exists():
        return

    yield from sorted(docs_path.rglob("*.md"))
