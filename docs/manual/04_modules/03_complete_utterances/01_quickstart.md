# Complete Utterances Module Quickstart

The Complete Utterances module supports CU coding workflows: creating coding workbooks, evaluating reliability, reselecting reliability samples, analyzing completed coding, and calculating rates.

Use `cus` in CLI commands; use "Complete Utterances" in prose and reports.

## Commands

| Command | Main use |
|---|---|
| `diaad cus files` | Create Complete Utterance coding and reliability workbooks. |
| `diaad cus evaluate` | Evaluate CU reliability. |
| `diaad cus reselect` | Select replacement CU reliability rows or samples. |
| `diaad cus analyze` | Summarize completed CU coding. |
| `diaad cus rates` | Calculate CU rates from analysis output and speaking-time values. |

## Typical Sequence

```text
transcripts tabularize
cus files
manual CU coding
cus evaluate
cus analyze
templates times
cus rates
```

Reliability and rates are optional workflow components, but they are common in research workflows.

## Common Outputs

| Step | Typical outputs |
|---|---|
| File generation | `cu_coding/cu_coding.xlsx`, `cu_reliability_coding.xlsx`, optional blind codebook |
| Reliability evaluation | `cu_reliability/` workbooks and report |
| Analysis | `cu_coding_analysis/cu_coding_by_utterance.xlsx`, `cu_coding_by_sample_long.xlsx`, `cu_coding_by_sample.xlsx` |
| Rates | `cu_coding_analysis/cu_coding_rates.xlsx` |

## Read Next

- Transcript tabularization: `docs/manual/03_features/01_transcript_tabularization.md`
- Configuration: `docs/manual/02_operation/04_configuration.md`
- Functional overview: `docs/manual/01_overview/04_functional_overview.md`

Later command pages describe coding columns, coder assignment, reliability pairing, and analysis output structure.
