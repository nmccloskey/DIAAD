# Utterance Template Example

This example demonstrates how `diaad templates utterances` creates blank utterance-level coding workbooks from transcript tables.

## Command

```bash
diaad templates utterances --config config
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
          utterance_coding_template.xlsx
          utterance_reliability_template.xlsx
          utterance_template_codebook.xlsx
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
stimulus_field: stimulus
```

## Advanced Config

```yaml
auto_blind: true
blind_cols:
- sample_id
metadata_source: transcript_tables
codebook_filename: ''
```

## Input Snippet

The command uses `diaad_data/input/transcript_tables/transcript_tables.xlsx`. The preview below is from the generated utterance coding template.

## Output Preview

`expected_outputs/templates_module/templates_utterances/utterance_coding_template.xlsx`

### Sheet: coding_template

| sample_id | utterance_id | coder_id | stimulus | utterance |
| --- | --- | --- | --- | --- |
| 1 | U0001 | 1 | picnic | Please tell the picnic story again. |
| 1 | U0002 | 1 | picnic | The family brought food to the park. |
| 1 | U0003 | 1 | picnic | The little girl [/] the little girl pours juice. |
| 1 | U0004 | 1 | picnic | Then they share sandwiches. |
| 1 | U0005 | 1 | picnic | Anything else? |
| 1 | U0006 | 1 | picnic | Yes, the dog waits beside them. |
| 1 | U0007 | 1 | picnic | The day is quiet. |
| 3 | U0001 | 1 | picnic | Tell me what is happening in the picnic picture. |

`expected_outputs/templates_module/templates_utterances/utterance_reliability_template.xlsx`

### Sheet: coding_template

| sample_id | utterance_id | coder_id | stimulus | utterance |
| --- | --- | --- | --- | --- |
| 1 | U0001 | 2 | picnic | Please tell the picnic story again. |
| 1 | U0002 | 2 | picnic | The family brought food to the park. |
| 1 | U0003 | 2 | picnic | The little girl [/] the little girl pours juice. |
| 1 | U0004 | 2 | picnic | Then they share sandwiches. |
| 1 | U0005 | 2 | picnic | Anything else? |
| 1 | U0006 | 2 | picnic | Yes, the dog waits beside them. |
| 1 | U0007 | 2 | picnic | The day is quiet. |
| 2 | U0001 | 1 | picnic | What do you notice first? |

`expected_outputs/templates_module/templates_utterances/utterance_template_codebook.xlsx`

### Sheet: Sheet1

| column | raw_value | blind_code |
| --- | --- | --- |
| sample_id | S001 | 1 |
| sample_id | S003 | 2 |
| sample_id | S002 | 3 |

## Notes

The primary and reliability workbooks are synthetic blank coding materials. The codebook maps blinded sample identifiers back to internal sample IDs.
