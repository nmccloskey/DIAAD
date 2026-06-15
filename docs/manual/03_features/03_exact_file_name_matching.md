# Exact File Name Matching

DIAAD treats configured file names as part of the analysis contract. Many commands do not simply look for "some spreadsheet that seems close." They look for one expected file, often by exact configured filename, and they stop when the file is missing or ambiguous.

This strictness is intentional. DIAAD workflows commonly move outputs from one run into later command inputs. Exact matching helps prevent a later run from silently reading an old workbook, a duplicate workbook, or a file with the right contents but the wrong provenance.

## The Basic Rule

For many important inputs, DIAAD searches recursively inside the configured input directory and the current run output directory, then requires exactly one match.

If no file is found, DIAAD raises a file-not-found error that lists the searched directories and the expected configured filename. If multiple matching files are found, DIAAD raises a multiple-files error that lists the matched paths and asks the user to remove duplicates, rename files, or configure a more specific filename.

Direct file paths are also accepted in exact-match paths when the configured value points to an existing file.

## Common Configured Filenames

These defaults come from `src/diaad/config/default_config.yaml`.

| Configuration field | Default filename | Typical use |
|---|---|---|
| `advanced.transcript_table_filename` | `transcript_tables.xlsx` | Transcript-table input for transcript-based workflows. |
| `advanced.cu_samples_filename` | `cu_coding_by_sample_long.xlsx` | Complete Utterance sample-level input for CU rates. |
| `advanced.cu_utts_filename` | `cu_coding_by_utterance.xlsx` | Complete Utterance utterance-level input for related workflows. |
| `advanced.word_count_filename` | `word_counting.xlsx` | Completed word-count coding workbook. |
| `advanced.wc_samples_filename` | `word_counting_by_sample.xlsx` | Sample-level word-count analysis workbook for rates. |
| `advanced.speaking_time_filename` | `speaking_times.xlsx` | Speaking-time workbook for rate calculations. |
| `advanced.powers_coding_filename` | `powers_coding.xlsx` | Completed POWERS coding workbook. |
| `advanced.powers_reliability_filename` | `powers_reliability_coding.xlsx` | POWERS reliability coding workbook. |
| `advanced.dct_coding_filename` | `conversation_turns.xlsx` | Completed Digital Conversational Turns coding workbook. |
| `advanced.dct_coding_reliability` | `conversation_turns_reliability.xlsx` | Digital Conversational Turns reliability coding workbook. |
| `advanced.metadata_source` | `transcript_tables.xlsx` | Metadata source for workflows that read metadata independently. |

The configured identifier columns are also part of the contract. The defaults are `sample_id` for `advanced.sample_id_column` and `utterance_id` for `advanced.utterance_id_column`.

## Transcript Tables

Transcript-table discovery is especially strict. Commands that require transcript tables search for the configured transcript table filename, which defaults to `transcript_tables.xlsx`.

If a command requires transcript tables and none are available, DIAAD does not guess. It asks the user to provide transcript tables, run `diaad transcripts tabularize` first, or set `project.auto_tabularize: true` so DIAAD can create transcript tables from input CHAT files when possible.

`auto_tabularize` defaults to `false`, so the ordinary workflow remains explicit and inspectable.

## Not Every Search Is Identical

Exact filename matching is a major pattern, but it is not the only discovery behavior in the codebase.

Examples of other behaviors include:

- CHAT input discovery recursively reads `.cha` files under the input directory while excluding configured reliability directories where appropriate.
- Target Vocabulary Coverage prefers an `unblind_utterance_data*.xlsx` input when present, then falls back to the configured transcript table filename.
- Target Vocabulary Coverage rates look for one or more `target_vocab_data_*.xlsx` analysis workbooks.
- Some template/subset workflows intentionally accept exactly one workbook by extension rather than one configured filename.
- Custom target-vocabulary resources are loaded from a configured resource path, which may point to project-specific JSON resources.

The shared idea is not that every command uses the same search. The shared idea is that when DIAAD needs an unambiguous analysis input, it avoids quiet guessing.

## Web App Uploads

The web app runs DIAAD in a temporary workspace. Uploaded input folders keep their nested relative paths under the web input directory, and uploaded config files are converted into the web run's temporary input and output paths.

This does not remove the filename contract. If a web run needs `transcript_tables.xlsx`, `word_counting.xlsx`, or another configured file, the uploaded files still need to match the configured names and expected folder structure closely enough for the command to find one intended input.

## Practical Habits

- Keep generated outputs in their timestamped output folders.
- When moving an output into a later input folder, preserve the configured filename unless you also update config.
- Avoid leaving multiple copies of the same expected workbook under the active input tree.
- Use `--dry-run-config` after changing filename or identifier settings.
- Treat file renaming as an analysis decision, not just a file-management cleanup.

## Read Next

- Configuration: `docs/manual/02_operation/04_configuration.md`
- Command-line operation: `docs/manual/02_operation/02_command_line.md`
- Web app operation: `docs/manual/02_operation/03_webapp.md`
- Transcript tabularization: `docs/manual/03_features/01_transcript_tabularization.md`

## Draft Review Notes

Before publication, review command-specific exceptions and wording for target-vocabulary discovery. The page should teach strictness without implying that every command uses identical matching logic.
