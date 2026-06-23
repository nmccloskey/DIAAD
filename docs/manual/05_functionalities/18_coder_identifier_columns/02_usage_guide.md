# Coder Identifier Columns Usage Guide

Coder identifiers are administrative labels in generated coding workbooks. They help project teams distribute coding work and keep primary and reliability assignments visible in the files that coders receive.

They are different from sample and utterance identifiers. `sample_id` and `utterance_id` define records and joins. `coder_id` identifies an assigned human coding role for a generated row.

## Configuration

Set the number of generated coder identifiers with:

```yaml
project:
  num_coders: 2
```

The packaged default is `0`, which means DIAAD still creates the relevant coder ID column where the workflow expects one, but leaves its values blank.

Generated command workflows use canonical labels:

| `num_coders` | General behavior |
|---|---|
| `0` | Coder ID cells are blank. |
| `1` | Generated rows are assigned to coder `1`. |
| `2` or more | Samples are split across available coder labels; reliability rows are assigned to an alternate coder when the workflow supports that pattern. |

The assignment happens by sample, not by individual utterance row. When a sample is assigned to a coder, all generated rows for that sample stay together.

## Module Patterns

| Area | Coder ID columns |
|---|---|
| Templates | `coder_id` |
| Complete Utterances, `num_coders` 0-2 | `coder_id` |
| Complete Utterances, three-coder schema | `coder1_id`, `coder2_id`, `coder3_id` |
| Word Counting | `coder_id` |
| POWERS | `coder_id` |

Complete Utterances caps the specialized three-coder schema at coder IDs `1`, `2`, and `3`. Values above `3` are reduced to that three-coder layout for CU coding files.

## Acceptable Values

Generated coder IDs are simple labels: blank, `1`, `2`, `3`, and so on depending on the workflow.

If a project edits coder identifiers manually, keep them simple and consistent. Whole-number labels, short alphanumeric codes, or stable initials are reasonable administrative values. Avoid using a coder ID column to store notes, scoring values, or multiple coders in one cell.

Do not change coder ID values midway through a distributed coding round unless the project is deliberately reassigning work. DIAAD does not treat the values as protected identifiers, so the project team is responsible for keeping assignments meaningful.

## Analysis Commands

Analysis commands do not require coder identifier columns. They use the substantive coding fields and configured record identifiers instead.

For example:

- `cus analyze` reads CU scoring columns such as `sv` and `rel`, or the configured CU paradigm columns.
- `words analyze` reads the configured word-count column.
- `powers analyze` reads POWERS coding fields such as turn type, speech-unit, and content-word fields.

This means `coder_id` is useful for managing the coding workflow, but it is not a variable that DIAAD summarizes as an outcome.

## Reliability Workbooks

Reliability workbooks use coder identifiers to make independent coding assignments visible. With multiple coders, DIAAD usually assigns selected reliability samples to a different coder label than the primary assignment when the workflow has enough coder labels to do so.

Reliability evaluation still compares primary and reliability records by the configured sample and utterance identifiers or by module-specific row keys. It does not require `coder_id` to be present.

## Legacy Column Name

Older DIAAD outputs and early draft examples used a generic `id` column for some coder assignments. Current user-facing outputs should use `coder_id`, or the CU three-coder names `coder1_id`, `coder2_id`, and `coder3_id`.

Some analysis cleanup paths still tolerate legacy `id` columns so older files can be read more easily. New files and documentation should use the clearer names.

## Read Next

- Complete Utterances files: `docs/manual/04_modules/03_complete_utterances/05_commands/01_files/02_usage_guide.md`
- Word Counting files: `docs/manual/04_modules/04_word_counting/05_commands/01_files/02_usage_guide.md`
- POWERS files: `docs/manual/04_modules/05_powers/05_commands/01_files/02_usage_guide.md`
- Template utterance files: `docs/manual/04_modules/02_templates/05_commands/01_utterances/02_usage_guide.md`
- Reliability selection, evaluation, and reselection: `docs/manual/05_functionalities/12_reliability_selection_evaluation_reselection/02_usage_guide.md`
