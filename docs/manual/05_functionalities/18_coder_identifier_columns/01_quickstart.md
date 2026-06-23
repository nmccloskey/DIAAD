# Coder Identifier Columns Quickstart

Coder identifier columns help organize manual coding workbooks. They show which coder a generated row is assigned to, especially when DIAAD splits samples across coders or creates a reliability workbook.

The main setting is:

```yaml
project:
  num_coders: 2
```

When `num_coders` is greater than zero, DIAAD generates coder labels such as `1`, `2`, and `3`. When it is `0`, generated coder ID cells are left blank.

## Where They Appear

Most coding-file generation workflows use one administrative column:

```text
coder_id
```

This applies to general templates, Word Counting files, and POWERS files.

Complete Utterances usually uses `coder_id` too. Its three-coder schema is the exception: primary files use `coder1_id` and `coder2_id`, and reliability files use `coder3_id` alongside the corresponding `c1_`, `c2_`, and `c3_` coding columns.

## What They Do Not Do

Coder identifier columns are not scoring columns. Analysis commands do not require them, and they are not used to compute CU, word-count, POWERS, or turn summaries.

If you receive a completed coding workbook without `coder_id`, you can usually still run the analysis command as long as the required sample, utterance, and coding columns are present.

## Read Next

- Coder identifier usage guide: `docs/manual/05_functionalities/18_coder_identifier_columns/02_usage_guide.md`
- Reliability selection, evaluation, and reselection: `docs/manual/05_functionalities/12_reliability_selection_evaluation_reselection/02_usage_guide.md`
- Configurable sample and utterance identifiers: `docs/manual/05_functionalities/10_configurable_sample_utterance_identifiers/02_usage_guide.md`
