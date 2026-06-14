# Target Vocabulary Coverage Module Quickstart

The Target Vocabulary Coverage module analyzes how much of a configured target vocabulary appears in transcript-derived language samples. It supports built-in resources, custom resources, resource checks, analysis, and rates.

## Commands

| Command | Main use |
|---|---|
| `diaad vocab file` | Create a blank custom target-vocabulary resource template. |
| `diaad vocab check` | Validate active built-in and custom resources. |
| `diaad vocab analyze` | Compute target vocabulary coverage. |
| `diaad vocab rates` | Calculate target-vocabulary rates from analysis output. |

## Typical Sequence

For built-in resources:

```text
transcripts tabularize
vocab check
vocab analyze
vocab rates
```

For custom resources:

```text
vocab file
edit resource JSON
vocab check
vocab analyze
vocab rates
```

## Common Outputs

| Step | Typical outputs |
|---|---|
| Resource template | `target_vocab/target_vocabulary_resource_template.json` |
| Resource check | `target_vocab/target_vocab_resource_check.txt` |
| Analysis | `target_vocab/target_vocab_data_YYMMDD_HHMM.xlsx` |
| Rates | `target_vocab/target_vocab_rates.xlsx` |

The analysis workbook contains `summary` and `details` sheets.

## Read Next

- Word Counting Versus Target Vocabulary Coverage: `docs/manual/03_features/02_word_counting_vs_target_vocabulary_coverage.md`
- Exact file name matching: `docs/manual/03_features/03_exact_file_name_matching.md`
- Configuration: `docs/manual/02_operation/04_configuration.md`

Later command and functionality pages describe resource JSON structure, custom resource paths, and norm handling.
