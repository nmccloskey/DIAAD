# Transcript Table Revision and CHAT Export Usage Guide

Transcript tables are designed to be a flexible but stable canonical representation for DIAAD-supported workflows. They are computer-friendly because they follow database logic, and human-editable because they are ordinary `.xlsx` workbooks.

## When To Revise A Transcript Table

Transcript table revision is useful when:

- transcript errors are found during coding;
- speaker labels need correction;
- metadata needs documented correction;
- an utterance needs insertion, deletion, split, or merge review;
- revised CHAT files are needed for sharing or another tool;
- the table is the best current representation of the transcript set.

Not every revision has the same consequence. Changing a typo before coding may be low risk. Changing utterance rows after manual coding may require recoding or reconciliation.

## Editing Text

For straightforward transcript corrections, edit the `utterance` cell and preserve the existing identifiers:

- keep `sample_id`;
- keep `utterance_id`;
- keep the original `position`;
- keep `position_sub` unless the row itself is being repositioned.

This tells downstream users that the analytic unit is the same row with corrected text.

## Inserting Utterances

Within each `sample_id`, the pair `(position, position_sub)` should be unique.

Use `position_sub = 0` for original skeleton rows. Use `position_sub > 0` for insertions. The original `position` values can stay as the skeleton even after edits; they do not need to become perfectly contiguous again.

Example original sequence:

| utterance_id | position | position_sub |
|---|---:|---:|
| `BUU4162` | 12 | 0 |
| `BUU4163` | 13 | 0 |

Insert a new utterance between them:

| utterance_id | position | position_sub |
|---|---:|---:|
| `BUU4162` | 12 | 0 |
| `BUU9999` | 12 | 1 |
| `BUU4163` | 13 | 0 |

If another utterance is inserted after `BUU4162`, use the next `position_sub`:

| utterance_id | position | position_sub |
|---|---:|---:|
| `BUU8888` | 12 | 2 |

DIAAD sorts by `position` and `position_sub` during CHAT export when those columns are present.

## Deleting Or Splitting Rows

If a row is deleted, document the reason outside the workbook or in a project note. Downstream files that refer to the deleted `utterance_id` may need review.

If an utterance is split into multiple rows, keep the original row only if it still represents the same analytic unit. New rows should receive new unique `utterance_id` values and explicit `position_sub` values.

## Metadata And Filenames

`transcripts chats` builds exported CHAT filenames from sample-level columns in the `samples` sheet. It excludes technical fields such as `input_order`, `shuffled_order`, and `derived_file`. If filenames would collide, DIAAD appends row indexes to affected names.

Before export, inspect the sample-level metadata columns that should appear in derived filenames.

## Optional CHAT Header

If a file matching this pattern is present under the input directory:

```text
*template_header.cha
```

DIAAD uses it as the exported CHAT header template. If no template is found, DIAAD uses a default header.

## Why Export CHAT Files

Export revised CHAT files when another workflow needs `.cha` files, such as:

- data sharing;
- archive preparation;
- AphasiaBank-style sharing;
- CLAN analysis;
- circulation of revised transcripts to collaborators.

Do not treat exported CHAT files as a lossless reconstruction of every original formatting detail. They are derived from the transcript table.

## Downstream Review

After transcript table revision, review downstream artifacts that may depend on the old table:

- coding files;
- reliability files;
- speaking-time files;
- target-vocabulary inputs;
- blinding codebooks;
- analysis workbooks;
- exported statistical tables.

The conservative pattern is to archive earlier outputs, regenerate affected files, recode or reconcile affected rows, and rerun analysis.

## Read Next

- `transcripts chats` usage guide: `docs/manual/04_modules/01_transcripts/05_commands/02_chats/02_usage_guide.md`
- Revision handling: `docs/manual/05_functionalities/11_revision_handling/02_usage_guide.md`
- Configurable identifiers: `docs/manual/05_functionalities/10_configurable_sample_utterance_identifiers/02_usage_guide.md`
- Run provenance: `docs/manual/05_functionalities/03_run_provenance_audit_artifacts/02_usage_guide.md`
