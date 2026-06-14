# `cus evaluate` Quickstart

`diaad cus evaluate` compares a completed CU coding workbook with a completed CU reliability workbook and writes utterance-level, sample-level, and report outputs.

## Run

```bash
diaad cus evaluate --config config
```

## Minimum Inputs

Provide completed workbooks named:

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

## Primary Outputs

By default, the command writes:

```text
cu_reliability/
  cu_reliability_coding_by_utterance.xlsx
  cu_reliability_coding_by_sample.xlsx
  cu_reliability_coding_report.txt
```

If multiple CU paradigms are configured, DIAAD writes paradigm-specific outputs under `cu_reliability/<PARADIGM>/`.

## Immediate Next Step

Review the report and the sample-level workbook together. Use the utterance-level workbook to inspect disagreements or unexpectedly low agreement for specific rows.

## Read Next

- Complete Utterances research context: `docs/manual/04_modules/03_complete_utterances/03_research_context.md`
- `cus files` usage: `docs/manual/04_modules/03_complete_utterances/05_commands/01_files/02_usage_guide.md`
- Generated Example I/O: `docs/manual/03_features/04_generated_example_io.md`
