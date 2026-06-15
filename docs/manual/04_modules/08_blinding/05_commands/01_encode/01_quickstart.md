# `blinding encode` Quickstart

`diaad blinding encode` masks configured identifier columns in one workbook and writes a reusable blind codebook.

## Run

```bash
diaad blinding encode --config config
```

## Minimum Inputs

Place one target `.xlsx` workbook in the configured input directory. If a compatible blind codebook already exists, place it there too.

The standalone command uses configured `advanced.blind_columns`, even when `advanced.auto_blind` is false. `auto_blind` controls supported workflows that apply blinding internally.

## Primary Outputs

By default, the command writes:

```text
blinding/
  <target_stem>_blinded.xlsx
  <target_stem>_blinding_diagnostics.xlsx
  blind_codebook.xlsx
```

The blinded workbook removes the raw configured columns and keeps suffixed blinded columns such as:

```text
sample_id_blinded
```

## Recommended Timing

Encode before manual coding when coders should work without raw sample identifiers. If a module's file-generation command supports `auto_blind`, that internal route can also produce coder-facing blinded files.

## Immediate Next Step

Inspect the diagnostics workbook and store `blind_codebook.xlsx` separately from coder-facing materials. Then distribute only the blinded workbook for manual coding.

## Read Next

- `blinding encode` usage guide: `docs/manual/04_modules/08_blinding/05_commands/01_encode/02_usage_guide.md`
- Blinding research context: `docs/manual/04_modules/08_blinding/03_research_context.md`
- Configuration: `docs/manual/02_operation/04_configuration.md`
- Generated Example I/O: `docs/manual/03_features/04_generated_example_io.md`
