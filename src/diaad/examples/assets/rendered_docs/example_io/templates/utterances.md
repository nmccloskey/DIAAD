# Utterance Template Example

This example demonstrates how `diaad templates utterances` creates blank utterance-level coding workbooks from transcript tables.

## Command

```bash
diaad templates utterances --config config
```

## Project Files

```
example_files/
  synthetic_project/
    README.md
    config/
      project.yaml
      advanced.yaml
    input/
      chat/
        P1_picnic_pre.cha
        P2_picnic_pre.cha
        P1_picnic_post.cha
        reliability/
          P1_picnic_pre.cha
          P2_picnic_pre.cha
      transcription_reliability_selection/
        transcription_reliability_samples.xlsx
    expected_outputs/
      templates_module/
        templates_utterances/
          utterance_coding_template.xlsx
          utterance_reliability_template.xlsx
          utterance_template_codebook.xlsx
```

## Basic Config

```yaml
input_dir: input
output_dir: output
reliability_fraction: 0.34
num_bins: 2
num_coders: 2
stimulus_field: stimulus
```

## Input Snippet

The command uses transcript tables created from the synthetic CHAT files. The preview below is from the generated utterance coding template.

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
