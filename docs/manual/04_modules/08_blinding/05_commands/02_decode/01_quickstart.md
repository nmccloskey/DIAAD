# `blinding decode` Quickstart

`diaad blinding decode` restores raw identifiers in one blinded workbook using a blind codebook.

## Run

```bash
diaad blinding decode --config config
```

## Minimum Inputs

Place one blinded `.xlsx` workbook and one compatible blind codebook in the configured input directory.

The codebook must include:

```text
column
raw_value
blind_code
```

## Primary Output

By default, the command writes:

```text
blinding/
  <target_stem>_decoded.xlsx
```

## Recommended Timing

Decode after manual coding and before DIAAD analysis when downstream commands need original sample identifiers, metadata joins, or canonical project IDs.

## Immediate Next Step

Inspect the decoded workbook to confirm that sample identifiers are restored as expected, then use the decoded file as the analysis input for the relevant DIAAD module.

## Read Next

- `blinding decode` usage guide: `docs/manual/04_modules/08_blinding/05_commands/02_decode/02_usage_guide.md`
- Blinding quickstart: `docs/manual/04_modules/08_blinding/01_quickstart.md`
- Blinding research context: `docs/manual/04_modules/08_blinding/03_research_context.md`
- Configuration: `docs/manual/02_operation/04_configuration.md`
