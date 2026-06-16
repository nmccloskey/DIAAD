from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

from psair.core.logger import get_rel_path
from psair.metadata.discovery import find_matching_files as psair_find_matching_files


MatchMode = Literal["exact", "contains"]
DEFAULT_TRANSCRIPT_TABLE_FILENAME = "transcript_tables.xlsx"


class MultipleFilesFoundError(RuntimeError):
    """Raised when DIAAD needs exactly one file but finds multiple matches."""


def _coerce_directories(
    directories: Path | str | Iterable[Path | str] | None,
) -> list[Path]:
    if directories is None:
        return [Path.cwd()]
    if isinstance(directories, (str, Path)):
        return [Path(directories)]
    return [Path(directory) for directory in directories]


def _display_paths(paths: Iterable[Path]) -> str:
    return "\n".join(f"  - {get_rel_path(path)}" for path in paths)


def _plural_label(label: str) -> str:
    return f"{label[:-5]} files" if label.endswith(" file") else f"{label} files"


def _configured_text(configured_filename: str | Path | None) -> str:
    if configured_filename is None:
        return ""
    return f" for configured filename '{Path(configured_filename).name}'"


def require_one_file(
    matches: Iterable[str | Path],
    *,
    label: str,
    configured_filename: str | Path | None = None,
    directories: Path | str | Iterable[Path | str] | None = None,
) -> Path:
    """
    Return the only matched file, or raise a user-actionable discovery error.
    """
    paths = [Path(path) for path in matches]
    configured = _configured_text(configured_filename)
    searched_dirs = _coerce_directories(directories)

    if not paths:
        raise FileNotFoundError(
            f"DIAAD did not find a {label}{configured}.\n"
            f"Searched directories:\n{_display_paths(searched_dirs)}\n"
            f"Please place the file in one of these directories or configure the "
            "exact filename."
        )

    if len(paths) > 1:
        plural = _plural_label(label)
        raise MultipleFilesFoundError(
            f"DIAAD has detected multiple {plural}{configured}.\n"
            f"Searched directories:\n{_display_paths(searched_dirs)}\n"
            f"Matched paths:\n{_display_paths(paths)}\n"
            "Please remove duplicates, rename files, or configure a more "
            "specific filename."
        )

    return paths[0]


def _direct_file_match(filename: str | Path) -> Path | None:
    candidate = Path(filename).expanduser()
    has_directory = candidate.is_absolute() or candidate.parent != Path(".")
    if has_directory and candidate.exists() and candidate.is_file():
        return candidate
    return None


def find_one_matching_file(
    *,
    directories: Path | str | Iterable[Path | str] | None = None,
    filename: str | Path | None = None,
    search_base: str = "",
    search_ext: str = ".xlsx",
    match_mode: MatchMode = "exact",
    match_metadata_fields: Iterable[str] | None = None,
    label: str = "file",
) -> Path:
    """
    Find exactly one file using PSAIR discovery with DIAAD's stricter policy.

    Exact mode compares the configured value to the full filename while still
    searching recursively in the supplied directories.
    """
    if match_mode == "exact":
        if filename is None:
            raise ValueError("filename is required when match_mode='exact'.")
        direct_match = _direct_file_match(filename)
        if direct_match is not None:
            return direct_match
        configured_filename: str | Path | None = filename
    else:
        configured_filename = f"{search_base}*{search_ext}"

    matches = psair_find_matching_files(
        directories=directories,
        filename=filename,
        search_base=search_base,
        search_ext=search_ext,
        match_mode=match_mode,
        match_metadata_fields=match_metadata_fields,
        deduplicate=False,
    )
    return require_one_file(
        matches,
        label=label,
        configured_filename=configured_filename,
        directories=directories,
    )


def find_one_file_by_extension(
    *,
    directories: Path | str | Iterable[Path | str] | None = None,
    search_ext: str = ".xlsx",
    label: str = "file",
) -> Path:
    """
    Find exactly one file by extension across one or more directories.

    This supports workflows that intentionally accept any filename with a
    specific extension, while preserving DIAAD's strict one-file policy.
    """
    searched_dirs = _coerce_directories(directories)
    matches = find_files_by_extension(
        directories=searched_dirs,
        search_ext=search_ext,
    )

    return require_one_file(
        matches,
        label=label,
        configured_filename=f"*{_normalize_extension(search_ext)}",
        directories=searched_dirs,
    )


def _normalize_extension(search_ext: str) -> str:
    """
    Normalize an extension string to include the leading dot.
    """
    ext = str(search_ext)
    return ext if ext.startswith(".") else f".{ext}"


def find_files_by_extension(
    *,
    directories: Path | str | Iterable[Path | str] | None = None,
    search_ext: str = ".xlsx",
) -> list[Path]:
    """
    Find files by extension across one or more directories.

    Discovery is recursive, keeps duplicate filenames as distinct paths, and
    skips temporary Excel lock files beginning with ``~$``.
    """
    ext = _normalize_extension(search_ext)
    searched_dirs = _coerce_directories(directories)
    return psair_find_matching_files(
        directories=searched_dirs,
        search_base="?",
        search_ext=ext,
        match_mode="contains",
        deduplicate=False,
        ignore_excel_temp_files=True,
    )


def find_transcript_table(
    *,
    directories: Path | str | Iterable[Path | str] | None = None,
    filename: str | Path = DEFAULT_TRANSCRIPT_TABLE_FILENAME,
    required: bool = True,
    match_metadata_fields: Iterable[str] | None = None,
    label: str = "transcript table file",
) -> Path | None:
    """
    Find one transcript table using DIAAD's exact-filename discovery policy.

    Recursive discovery must resolve to at most one exact filename match. When
    ``required`` is false, a missing table returns None, but multiple matches
    still raise an actionable error.
    """
    try:
        return find_one_matching_file(
            directories=directories,
            filename=filename,
            match_mode="exact",
            match_metadata_fields=match_metadata_fields,
            label=label,
        )
    except FileNotFoundError:
        if required:
            raise
        return None
