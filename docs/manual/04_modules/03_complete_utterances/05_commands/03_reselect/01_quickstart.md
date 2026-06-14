# `cus reselect` Quickstart

`diaad cus reselect` creates a replacement CU reliability workbook while avoiding samples already present in prior CU reliability files.

## Run

```bash
diaad cus reselect --config config
```

## Minimum Inputs

Provide prior CU coding and reliability workbooks under the configured input directory:

```text
cu_coding.xlsx
cu_reliability_coding.xlsx
```

A common layout is:

```text
diaad_data/input/
  cu_coding/
    cu_coding.xlsx
    cu_reliability_coding.xlsx
```

## Primary Output

By default, the command writes:

```text
reselected_cu_coding_reliability/
  reselected_cu_reliability_coding.xlsx
```

## Immediate Next Step

Open the reselected workbook and confirm that selected sample IDs were not already used in the prior reliability workbook.

## Read Next

- `cus evaluate` usage: `docs/manual/04_modules/03_complete_utterances/05_commands/02_evaluate/02_usage_guide.md`
- Configuration: `docs/manual/02_operation/04_configuration.md`
- Generated Example I/O: `docs/manual/03_features/04_generated_example_io.md`
