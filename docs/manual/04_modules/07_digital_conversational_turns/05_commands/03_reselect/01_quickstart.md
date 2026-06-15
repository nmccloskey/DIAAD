# `turns reselect` Quickstart

`diaad turns reselect` selects replacement samples for DCT reliability coding.

## Run

```bash
diaad turns reselect --config config
```

## Minimum Inputs

Place prior DCT coding and reliability workbooks under the configured input directory. The command searches recursively for files matching:

```text
*conversation_turns_template.xlsx
*conversation_turns_reliability*.xlsx
```

The primary workbook must include the configured sample identifier column.

## Primary Output

By default, the command writes:

```text
reselected_turns_reliability/
  reselected_conversation_turns_reliability_template.xlsx
```

The output keeps the selected sample rows and clears the `turns` cells for fresh reliability coding.

## Immediate Next Step

Give the reselected reliability workbook to a reliability coder, then evaluate it with `diaad turns evaluate` after coding is complete.

## Read Next

- `turns reselect` usage guide: `docs/manual/04_modules/07_digital_conversational_turns/05_commands/03_reselect/02_usage_guide.md`
- `turns evaluate` quickstart: `docs/manual/04_modules/07_digital_conversational_turns/05_commands/02_evaluate/01_quickstart.md`
- Configuration: `docs/manual/02_operation/04_configuration.md`
- Exact file-name matching: `docs/manual/03_features/03_exact_file_name_matching.md`
