# Transcription Reliability Evaluation Example

This example demonstrates how `diaad transcripts evaluate` compares original CHAT files with synthetic reliability transcriptions.

## Command

```bash
diaad transcripts evaluate --config config
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
        reliability/
          P1_picnic_pre.cha
          P2_picnic_pre.cha
    output/
      diaad_YYMMDD_HHMM/
        transcription_reliability_evaluation/
          transcription_reliability_evaluation.xlsx
          transcription_reliability_report.txt
          global_alignments/
        logs/
          diaad_YYMMDD_HHMM.log
          diaad_YYMMDD_HHMM_metadata.json
```

## Basic Config

```yaml
input_dir: diaad_data/input
output_dir: diaad_data/output
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
reliability_tag: _reliability
reliability_dirname: reliability
```

## Input Snippet

`diaad_data/input/chat/reliability/P1_picnic_pre.cha`

```text
@Begin
@Languages:	eng
@Participants:	PAR Participant, INV Investigator
@ID:	eng|synthetic|PAR|||||Participant|||
@ID:	eng|synthetic|INV|||||Investigator|||
@Date:	01-JAN-2026
@Comment:	Fully synthetic DIAAD reliability example.
*INV:	Tell me what is happening in the picnic picture.
*PAR:	The family is sitting on the blanket.
*PAR:	They have sandwiches, apples, and juice.
*INV:	What is the child doing?
*PAR:	She is pouring juice.
```

## Output Preview

`expected_outputs/transcripts_module/transcripts_evaluate/transcription_reliability_evaluation.xlsx`

| participant_id | stimulus | timepoint | original_file | reliability_file | org_num_tokens | rel_num_tokens | perc_diff_num_tokens | org_num_chars | rel_num_chars | perc_diff_num_chars | levenshtein_distance | levenshtein_similarity | needleman_wunsch_score | needleman_wunsch_norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | picnic | pre | P1_picnic_pre.cha | P1_picnic_pre_reliability.cha | 24 | 23 | 4.25531914893617 | 136 | 136 | 0.0 | 9 | 0.9338235294117647 | 125 | 0.9191176470588235 |
| P2 | picnic | pre | P2_picnic_pre.cha | P2_picnic_pre_reliability.cha | 22 | 22 | 0.0 | 111 | 114 | 2.666666666666667 | 6 | 0.9473684210526316 | 105 | 0.9210526315789473 |

`expected_outputs/transcripts_module/transcripts_evaluate/transcription_reliability_report.txt`

```text
Transcription Reliability Report
================================
Number of samples: 2

Levenshtein similarity score summary stats:
  • Average: 0.941
  • Standard Deviation: 0.010
  • Min: 0.934
  • Max: 0.947
```

## Notes

Reliability transcripts are believable synthetic variants of the original examples. DIAAD matches originals and reliability files by configured metadata fields.
