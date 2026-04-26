# Transcription Reliability Selection Example

This example demonstrates how `diaad transcripts select` selects synthetic CHAT files for secondary transcription and writes blank reliability templates.

## Command

```bash
diaad transcripts select --config config
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
      transcripts_select/
        P1_picnic_pre_reliability.cha
        P2_picnic_pre_reliability.cha
        transcription_reliability_samples.xlsx
```

## Basic Config

```yaml
input_dir: input
output_dir: output
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

The command uses the synthetic CHAT files in `input/chat/`.

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

`expected_outputs/transcripts_select/transcription_reliability_samples.xlsx`

### Sheet: reliability_selection

| file | file_ext | file_dir | participant_id | stimulus | timepoint |
| --- | --- | --- | --- | --- | --- |
| P1_picnic_pre | .cha | input/chat | P1 | picnic | pre |
| P2_picnic_pre | .cha | input/chat | P2 | picnic | pre |

### Sheet: all_transcripts

| file | file_ext | file_dir | participant_id | stimulus | timepoint | selected_for_reliability |
| --- | --- | --- | --- | --- | --- | --- |
| P1_picnic_post | .cha | input/chat | P1 | picnic | post | 0 |
| P1_picnic_pre | .cha | input/chat | P1 | picnic | pre | 1 |
| P2_picnic_pre | .cha | input/chat | P2 | picnic | pre | 1 |

## Notes

The blank reliability `.cha` files contain CHAT headers only. They are generated artifacts for transcription workflow setup, not completed reliability transcripts.
