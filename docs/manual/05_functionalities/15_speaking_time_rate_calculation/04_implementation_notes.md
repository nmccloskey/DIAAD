# Speaking-Time Rate Calculation Implementation Notes

Speaking-time templates and rate calculations use a shared utility layer plus module-specific wrappers.

## Source Anchors

Primary sources:

- `src/diaad/coding/templates/times.py`
- `src/diaad/coding/utils/rates.py`
- `src/diaad/coding/compl_utts/rates.py`
- `src/diaad/coding/word_counts/rates.py`
- `src/diaad/coding/powers/rates.py`
- `src/diaad/coding/target_vocab/rates.py`
- `src/diaad/coding/target_vocab/analysis.py`
- `src/diaad/core/config.py`
- `src/diaad/core/run_context.py`

Relevant tests:

- `tests/test_coding/test_utils/test_rates.py`
- module-specific rate tests where present

## Template Generation

`build_speaking_time_template()` reads the transcript table `samples` sheet, keeps one row per configured sample identifier, and adds a blank `speaking_time` column.

`make_speaking_time_template_files()` writes:

```text
coding_templates/speaking_times.xlsx
```

The default speaking-time filename is configured as:

```yaml
advanced:
  speaking_time_filename: speaking_times.xlsx
  speaking_time_column: speaking_time
```

## Shared Rate Utilities

`read_speaking_time_table()` discovers the configured speaking-time file, requires the configured sample identifier and speaking-time columns, coerces speaking time to numeric, sums duplicate sample IDs with a warning, and adds:

```text
speaking_minutes
```

`compute_rate_per_minute()` returns missing values when the numerator is missing, speaking minutes are missing, or speaking minutes are less than or equal to zero.

`add_rate_columns()` creates one new rate column per numerator using the `_per_min` suffix.

## Module Wrappers

`calculate_cu_rates()` reads `cu_coding_by_sample_long.xlsx`, merges speaking time, and writes `cu_coding_rates.xlsx` under `cu_coding_analysis/`.

`calculate_word_count_rates()` reads `word_counting_by_sample.xlsx`, merges speaking time, and writes `word_counting_rates.xlsx` under `word_count_analysis/`.

`calculate_powers_rates()` reads POWERS analysis workbooks matching `*powers*analysis*.xlsx`, merges speaking time, infers numeric count-like columns from the `Dialogs` sheet, skips proportions and ratios, and writes `powers_coding_rates.xlsx` under `powers_coding_analysis/`.

`calculate_target_vocab_rates()` reads `summary` sheets from workbooks matching `target_vocab_data_*.xlsx`, requires `speaking_time`, infers eligible numeric count-like columns, and writes `target_vocab_rates.xlsx` under `target_vocab/`.

## Boundary

DIAAD does not calculate speaking time from media files, CHAT timestamps, or transcript text. It reads user-entered speaking-time values and applies shared per-minute arithmetic.

## Read Next

- `templates times` implementation notes: `docs/manual/04_modules/02_templates/05_commands/03_times/04_implementation_notes.md`
- Rate command pages: `docs/manual/04_modules/`
- Testing: `docs/manual/02_operation/05_testing.md`
