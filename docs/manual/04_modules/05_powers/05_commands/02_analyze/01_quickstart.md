# `powers analyze` Quickstart

`diaad powers analyze` summarizes completed utterance-level POWERS coding into utterance, turn, speaker, and dialog sheets.

## Run

```bash
diaad powers analyze --config config
```

## Minimum Input

Provide a completed primary POWERS coding workbook. By default, DIAAD looks for:

```text
powers_coding.xlsx
```

A common input layout is:

```text
diaad_data/input/
  powers_coding/
    powers_coding.xlsx
```

## Primary Output

By default, the command writes:

```text
powers_coding_analysis/
  powers_analysis.xlsx
```

## Immediate Next Step

Open `powers_analysis.xlsx` and inspect the `Dialogs`, `Speakers`, and `Turns` sheets for implausible totals, missing numeric coding, and unexpected turn labels.

## Read Next

- `powers files` usage guide: `docs/manual/04_modules/05_powers/05_commands/01_files/02_usage_guide.md`
- `powers rates` quickstart: `docs/manual/04_modules/05_powers/05_commands/03_rates/01_quickstart.md`
- POWERS module quickstart: `docs/manual/04_modules/05_powers/01_quickstart.md`
- Generated Example I/O: `docs/manual/03_features/04_generated_example_io.md`
