# `cus analyze` Quickstart

`diaad cus analyze` summarizes a completed CU coding workbook by utterance and by sample.

## Run

```bash
diaad cus analyze --config config
```

## Minimum Inputs

Provide a completed workbook named:

```text
cu_coding.xlsx
```

A common layout is:

```text
diaad_data/input/
  cu_coding/
    cu_coding.xlsx
    cu_blind_codebook.xlsx
```

The codebook is only needed when the coding workbook uses blinded identifiers and analysis should reconnect or preserve the intended identifier mapping.

## Primary Outputs

By default, the command writes:

```text
cu_coding_analysis/
  cu_coding_by_utterance.xlsx
  cu_coding_by_sample_long.xlsx
  cu_coding_by_sample.xlsx
```

With analysis-stage blinding, DIAAD may also write an analysis blind codebook and diagnostics workbook.

## Immediate Next Step

Inspect the long sample summary first. It is the canonical input for `diaad cus rates`.

## Read Next

- Complete Utterances research context: `docs/manual/04_modules/03_complete_utterances/03_research_context.md`
- `cus rates` usage: `docs/manual/04_modules/03_complete_utterances/05_commands/05_rates/02_usage_guide.md`
- Generated Example I/O: `docs/manual/03_features/04_generated_example_io.md`
