# Configured Filenames, File Discovery, and Input Selection Quickstart

DIAAD uses strict file discovery for many commands. When a command needs one input workbook, DIAAD usually expects exactly one matching file with the configured filename.

## The Practical Rule

Keep configured filenames stable and avoid duplicate matching files in the input tree.

Common defaults include:

```yaml
advanced:
  transcript_table_filename: transcript_tables.xlsx
  cu_samples_filename: cu_coding_by_sample_long.xlsx
  cu_utts_filename: cu_coding_by_utterance.xlsx
  word_count_filename: word_counting.xlsx
  powers_coding_filename: powers_coding.xlsx
  dct_coding_filename: conversation_turns.xlsx
```

If a command reports that no file or multiple files were found, fix the input directory or the configured filename before rerunning.

## Read Next

- Exact file name matching feature: `docs/manual/03_features/03_exact_file_name_matching.md`
- Configuration: `docs/manual/02_operation/04_configuration.md`
- Command-line operation: `docs/manual/02_operation/02_command_line.md`
