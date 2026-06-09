# Conversation Turns File Example

This example demonstrates how `diaad turns files` creates blank digital conversation-turn coding and reliability workbooks.

## Command

```bash
diaad turns files --config config
```

## Project Files

```
your_project/
  config/
    project.yaml
    advanced.yaml
  diaad_data/
    input/
      transcript_tables/
        transcript_tables.xlsx
    output/
      diaad_YYMMDD_HHMM/
        coding_templates/
          conversation_turns_template.xlsx
          conversation_turns_reliability_template.xlsx
          conversation_turns_template_codebook.xlsx
        logs/
          diaad_YYMMDD_HHMM.log
          diaad_YYMMDD_HHMM_metadata.json
```

## Basic Config

```yaml
input_dir: diaad_data/input
output_dir: diaad_data/output
reliability_fraction: 0.34
num_bins: 2
num_coders: 2
metadata_fields:
  participant_id: P\d+
  stimulus:
  - picnic
  timepoint:
  - pre
  - post
```

## Advanced Config

```yaml
auto_blind: true
blind_columns:
- sample_id
metadata_source: transcript_tables.xlsx
codebook_filename: ''
```

## Input Snippet

The command uses `diaad_data/input/transcript_tables/transcript_tables.xlsx` to create one row per sample and bin.

## Output Preview

`expected_outputs/turns_module/turns_files/conversation_turns_template.xlsx`

| sample_id | coder_id | session | bin | turns |
| --- | --- | --- | --- | --- |
| 1 | 1 |  | 1 |  |
| 1 | 1 |  | 2 |  |
| 2 | 2 |  | 1 |  |
| 2 | 2 |  | 2 |  |
| 3 | 1 |  | 1 |  |
| 3 | 1 |  | 2 |  |

`expected_outputs/turns_module/turns_files/conversation_turns_reliability_template.xlsx`

| sample_id | coder_id | session | bin | turns |
| --- | --- | --- | --- | --- |
| 1 | 2 |  | 1 |  |
| 1 | 2 |  | 2 |  |
| 2 | 1 |  | 1 |  |
| 2 | 1 |  | 2 |  |

## Notes

The generated local example fills separate synthetic turn strings into conversation-turn workbooks so downstream examples can be demonstrated. Digits identify speakers and dot markers are preserved for analysis.
