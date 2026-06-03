from __future__ import annotations

from pathlib import Path
from typing import Iterator

from psair.examples import get_rendered_docs_path, iter_rendered_markdown_files


DOC_PACKAGE = "diaad.examples"
DOC_ROOT = ("assets", "rendered_docs", "example_io")


def get_example_io_docs_path() -> Path:
    """Return the packaged example I/O markdown directory."""
    return get_rendered_docs_path(DOC_PACKAGE, *DOC_ROOT)


def iter_example_io_markdown_files() -> Iterator[Path]:
    """Yield packaged example I/O markdown files in display order."""
    yield from iter_rendered_markdown_files(DOC_PACKAGE, *DOC_ROOT)
