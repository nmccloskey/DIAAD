# `powers evaluate` Quickstart

`diaad powers evaluate` compares completed primary and reliability POWERS coding workbooks.

## Run

```bash
diaad powers evaluate --config config
```

## Minimum Inputs

Provide completed primary and reliability workbooks:

```text
powers_coding.xlsx
powers_reliability_coding.xlsx
```

A common input layout is:

```text
diaad_data/input/
  powers_coding/
    powers_coding.xlsx
    powers_reliability_coding.xlsx
```

## Primary Outputs

By default, the command writes:

```text
powers_reliability/
  powers_reliability_results.xlsx
  powers_reliability_report.txt
```

## Immediate Next Step

Read the report for coverage and reliability summaries, then inspect `powers_reliability_results.xlsx` for metric-specific disagreements and low-variance warnings.

## Read Next

- `powers evaluate` research context: `docs/manual/04_modules/05_powers/05_commands/04_evaluate/03_research_context.md`
- POWERS module research context: `docs/manual/04_modules/05_powers/03_research_context.md`
- Exact file-name matching: `docs/manual/03_features/03_exact_file_name_matching.md`
- Configuration: `docs/manual/02_operation/04_configuration.md`
- Generated Example I/O: `docs/manual/03_features/04_generated_example_io.md`
