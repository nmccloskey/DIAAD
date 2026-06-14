# POWERS Module Quickstart

The POWERS module supports dialogue-oriented POWERS coding workflows: coding-file generation, automated first-pass support for selected fields, human review, reliability evaluation and reselection, analysis, and rates.

## Commands

| Command | Main use |
|---|---|
| `diaad powers files` | Create POWERS coding and reliability workbooks. |
| `diaad powers analyze` | Summarize completed POWERS coding. |
| `diaad powers rates` | Calculate POWERS rates from analysis output and speaking-time values. |
| `diaad powers evaluate` | Evaluate POWERS reliability. |
| `diaad powers reselect` | Select replacement POWERS reliability material. |

## Typical Sequence

```text
transcripts tabularize
powers files
human review of POWERS coding
powers evaluate
powers analyze
templates times
powers rates
```

Automation is support for a first pass. It is not a replacement for human coding.

## Common Outputs

| Step | Typical outputs |
|---|---|
| File generation | `powers_coding/powers_coding.xlsx`, `powers_reliability_coding.xlsx`, optional blind codebook |
| Analysis | `powers_coding_analysis/powers_analysis.xlsx` |
| Rates | `powers_coding_analysis/powers_coding_rates.xlsx` |
| Reliability | `powers_reliability/powers_reliability_results.xlsx`, report |

## Dependencies

POWERS automation uses NLP support. The default configured spaCy model is `en_core_web_sm`. Install DIAAD with the relevant NLP dependencies or disable automation if a project will fill all POWERS fields manually.

## Read Next

- Transcript tabularization: `docs/manual/03_features/01_transcript_tabularization.md`
- Installation: `docs/manual/02_operation/01_installation.md`
- Configuration: `docs/manual/02_operation/04_configuration.md`

Later POWERS command and automation-functionality pages will describe supported automated fields and workbook expectations in more detail.
