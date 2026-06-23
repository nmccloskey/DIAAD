# `cus files` Usage Guide

Use `diaad cus files` when a transcript-table project is ready for Complete Utterance coding.

## Before Running

Tabularize transcripts first, then inspect the transcript table. The command reads utterance-level transcript data and creates coding rows from those utterances.

Important configuration:

| Setting | Default | Effect |
|---|---|---|
| `project.reliability_fraction` | `0.2` | Fraction of samples selected for the reliability workbook. |
| `project.num_coders` | `0` | Determines coder assignment mode. |
| `project.random_seed` | `99` | Seed used for sample shuffling and reliability selection. |
| `project.exclude_speakers` | `[]` | Speaker labels treated as not applicable for CU coding and later analysis. |
| `project.stimulus_column` | `''` | Optional stimulus column to preserve from transcript data. |
| `advanced.cu_paradigms` | `[]` | Optional CU paradigm labels for paradigm-specific SV/REL columns. |
| `advanced.sample_id_column` | `sample_id` | Sample identifier column. |
| `advanced.utterance_id_column` | `utterance_id` | Utterance identifier column. |
| `advanced.auto_blind` | `false` | Whether supported coding exports should blind configured columns. |

## Coder Modes

`num_coders` controls the generated coding schema:

| `num_coders` | Behavior |
|---|---|
| `0` | Blank primary coding file with blank coder IDs. |
| `1` | Primary file uses coder ID `1`; reliability subset is blank rather than assigned to another coder. |
| `2` | Samples are split across coders `1` and `2`; reliability rows are assigned to the opposite coder. |
| `3` or more | Uses the three-coder schema with `c1` and `c2` primary columns and `c3` reliability columns. Values above `3` are capped to coder IDs `1`, `2`, and `3`. |

## Coding Columns

With no multi-paradigm configuration, the primary coding columns are:

```text
coder_id
sv
rel
comment
```

CU is later derived as positive only when both `sv` and `rel` are coded `1`.

When two or more `advanced.cu_paradigms` are configured, DIAAD creates paradigm-specific columns such as:

```text
sv_AAE
rel_AAE
```

For the three-coder schema, coder assignment columns are named `coder1_id`,
`coder2_id`, and `coder3_id` alongside the corresponding `c1_`, `c2_`, and
`c3_` coding fields.

The exact coding interpretation belongs to the project's CU protocol. DIAAD creates the workbook structure; it does not decide ambiguous CU cases.

## Blinding

When `advanced.auto_blind` is true, generated coding files can blind configured identifier columns and write `cu_blind_codebook.xlsx`. Protect the codebook because later analysis may need it to reconnect blinded identifiers.

## Common Problems

If the command cannot find transcript tables, run `diaad transcripts tabularize` first or check `advanced.transcript_table_filename`.

If coding columns are not the schema you expected, check `num_coders` and `advanced.cu_paradigms`.

If investigator or other non-target speaker rows should be excluded from coding, configure `project.exclude_speakers` before generating files.
