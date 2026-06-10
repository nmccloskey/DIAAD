# Transcription Reliability Selection Example

This example demonstrates how `diaad transcripts select` selects synthetic CHAT files for secondary transcription and writes blank reliability templates.

## Command

```bash
diaad transcripts select --config config
```

## Project Files

```
your_project/
  config/
    project.yaml
    advanced.yaml
  diaad_data/
    input/
      chat/
        P1_picnic_pre.cha
        P2_picnic_pre.cha
        P1_picnic_post.cha
    output/
      diaad_YYMMDD_HHMM/
        transcription_reliability_selection/
          P1_picnic_pre_reliability.cha
          P2_picnic_pre_reliability.cha
          transcription_reliability_samples.xlsx
        logs/
          diaad_YYMMDD_HHMM.log
          diaad_YYMMDD_HHMM_metadata.json
```

## Basic Config

```yaml
input_dir: diaad_data/input
output_dir: diaad_data/output
random_seed: 99
reliability_fraction: 0.34
metadata_fields:
  participant_id: P\d+
  stimulus:
  - picnic
  timepoint:
  - pre
  - post
```

## Input Snippet

The command uses the synthetic CHAT files in `diaad_data/input/chat/`.

```text
@Begin
@Languages:	eng
@Participants:	PAR Participant, INV Investigator
@ID:	eng|synthetic|PAR|||||Participant|||
@ID:	eng|synthetic|INV|||||Investigator|||
@Date:	01-JAN-2026
@Comment:	Fully synthetic DIAAD example.
*INV:	What do you notice first?
*PAR:	A picnic.
*PAR:	The dad is opening the basket.
*PAR:	The dog wants food!
```

## Output Preview

`expected_outputs/transcripts_module/transcripts_select/transcription_reliability_samples.xlsx`

### Sheet: reliability_selection

| sample_id | file | file_ext | file_dir | input_order | shuffled_order | participant_id | stimulus | timepoint | metadata_mismatch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S002 | P1_picnic_pre | .cha | input/chat | 2 |  | P1 | picnic | pre | 0 |
| S003 | P2_picnic_pre | .cha | input/chat | 3 |  | P2 | picnic | pre | 0 |

### Sheet: all_transcripts

| sample_id | file | file_ext | file_dir | input_order | shuffled_order | participant_id | stimulus | timepoint | metadata_mismatch | selected_for_reliability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S001 | P1_picnic_post | .cha | input/chat | 1 |  | P1 | picnic | post | 0 | 0 |
| S002 | P1_picnic_pre | .cha | input/chat | 2 |  | P1 | picnic | pre | 0 | 1 |
| S003 | P2_picnic_pre | .cha | input/chat | 3 |  | P2 | picnic | pre | 0 | 1 |

## Notes

The blank reliability `.cha` files contain CHAT headers only. They are generated artifacts for transcription workflow setup, not completed reliability transcripts.
