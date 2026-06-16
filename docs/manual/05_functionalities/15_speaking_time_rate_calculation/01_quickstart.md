# Speaking-Time Rate Calculation Quickstart

DIAAD rate commands convert count-like measures into per-minute rates using per-sample speaking time.

## Create A Speaking-Time Template

For most rate workflows, first create a speaking-time workbook:

```bash
diaad templates times
```

This writes:

```text
coding_templates/speaking_times.xlsx
```

Enter values in seconds in the `speaking_time` column.

## Run A Rate Command

Supported rate commands include:

```bash
diaad cus rates
diaad words rates
diaad powers rates
diaad vocab rates
```

Rate outputs are module-specific, but the shared rule is:

```text
rate = count-like value / speaking_minutes
```

where:

```text
speaking_minutes = speaking_time / 60
```

## Read Next

- `templates times` command: `docs/manual/04_modules/02_templates/05_commands/03_times/01_quickstart.md`
- Complete Utterances rates: `docs/manual/04_modules/03_complete_utterances/05_commands/05_rates/01_quickstart.md`
- Word Counting rates: `docs/manual/04_modules/04_word_counting/05_commands/05_rates/01_quickstart.md`
- POWERS rates: `docs/manual/04_modules/05_powers/05_commands/03_rates/01_quickstart.md`
- Target Vocabulary Coverage rates: `docs/manual/04_modules/06_target_vocabulary_coverage/05_commands/04_rates/01_quickstart.md`
