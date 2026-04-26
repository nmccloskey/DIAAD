# Transcription Reliability Reselection Example

This example demonstrates how `diaad transcripts reselect` chooses replacement reliability samples after an earlier selection has already been used.

## Command

```bash
diaad transcripts reselect --config config
```

## Project Files

```
your_project/
  config/
    project.yaml
    advanced.yaml
  diaad_data/
    input/
      transcription_reliability_selection/
        transcription_reliability_samples.xlsx
    output/
      diaad_YYMMDD_HHMM/
        reselected_transcription_reliability/
          reselected_transcription_reliability_samples.xlsx
        logs/
          diaad_YYMMDD_HHMM.log
          diaad_YYMMDD_HHMM_metadata.json
```

## Basic Config

```yaml
input_dir: diaad_data/input
output_dir: diaad_data/output
reliability_fraction: 0.34
```

## Input Snippet

The reselection command reads the prior selection workbook:

`diaad_data/input/transcription_reliability_selection/transcription_reliability_samples.xlsx`

## Output Preview

`expected_outputs/transcripts_module/transcripts_reselect/reselected_transcription_reliability/reselected_transcription_reliability_samples.xlsx`

### Sheet: reselected_reliability

| file | file_ext | file_dir | participant_id | stimulus | timepoint | selected_for_reliability |
| --- | --- | --- | --- | --- | --- | --- |
| P1_picnic_post | .cha | input/chat | P1 | picnic | post | 1 |

## Notes

The synthetic project has three samples. Because two are already selected in the first reliability pass, only one unused candidate remains for reselection.
