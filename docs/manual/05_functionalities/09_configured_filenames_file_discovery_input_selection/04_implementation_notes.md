# Configured Filenames, File Discovery, and Input Selection Implementation Notes

Configured filename discovery is implemented in DIAAD metadata discovery helpers and then reused by transcript, coding, blinding, and analysis modules.

## Source Anchors

Primary sources:

- `src/diaad/metadata/discovery.py`
- `src/diaad/core/config.py`
- `src/diaad/core/run_context.py`
- module-specific file-generation and analysis modules under `src/diaad/coding/`

Relevant tests:

- `tests/test_metadata/test_utils.py`
- `tests/test_transcripts/test_detabularization.py`
- module-specific filename and identifier tests

## Strict One-File Policy

`require_one_file()` enforces DIAAD's one-file policy. It raises:

- `FileNotFoundError` when no matching file is found;
- `MultipleFilesFoundError` when more than one matching file is found.

The error messages include searched directories and, for duplicate matches, the matched paths.

## Exact Filename Discovery

`find_one_matching_file()` supports exact and contains-style matching. In exact mode, a configured `filename` is required. The configured value is compared to the full filename during recursive discovery.

If the configured filename is a direct path to an existing file, DIAAD returns that file without recursive search.

`find_transcript_table()` wraps exact matching for transcript tables. When `required=False`, a missing transcript table returns `None`, but duplicate matches still raise an error.

## Extension-Only Discovery

`find_one_file_by_extension()` supports workflows that intentionally accept any workbook with a given extension. It still requires exactly one matching file and ignores temporary Excel lock files whose names begin with `~$`.

## Run Context Search

For transcript tables, `RunContext.find_transcript_tables()` searches the configured input directory and the current run output directory for the configured transcript table filename. This lets a multi-command run use a table created earlier in the same run.

## Read Next

- Exact file name matching feature: `docs/manual/03_features/03_exact_file_name_matching.md`
- Configuration implementation notes: `docs/manual/05_functionalities/01_configuration_sources_defaults_overrides/04_implementation_notes.md`
- Transcript preprocessing implementation notes: `docs/manual/05_functionalities/06_transcript_preprocessing_tabularization_chat_export/04_implementation_notes.md`
