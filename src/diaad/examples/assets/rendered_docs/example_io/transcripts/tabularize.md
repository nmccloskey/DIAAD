# Transcript Tabularization Example

This example demonstrates how `diaad transcripts tabularize` converts tiny synthetic CHAT files into sample- and utterance-level workbook sheets.

## Command

```bash
diaad transcripts tabularize --config config
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
        transcript_tables/
          transcript_tables.xlsx
        logs/
          diaad_YYMMDD_HHMM.log
          diaad_YYMMDD_HHMM_metadata.json
```

## Basic Config

```yaml
input_dir: diaad_data/input
output_dir: diaad_data/output
random_seed: 99
shuffle_samples: false
metadata_fields:
  participant_id: P\d+
  stimulus:
  - picnic
  timepoint:
  - pre
  - post
```

## Input Snippet

`diaad_data/input/chat/P1_picnic_pre.cha`

```text
@Begin
@Languages:	eng
@Participants:	PAR Participant, INV Investigator
@ID:	eng|synthetic|PAR|||||Participant|||
@ID:	eng|synthetic|INV|||||Investigator|||
@Date:	01-JAN-2026
@Comment:	Fully synthetic DIAAD example.
*INV:	Tell me what is happening in the picnic picture.
*PAR:	The family is sitting on a blanket.
*PAR:	They have sandwiches, apples, and juice.
*INV:	What is the child doing?
*PAR:	She is pouring a drink.
```

## Output Preview

`expected_outputs/transcripts_module/transcripts_tabularize/transcript_table.xlsx`

### Sheet: samples

| sample_id | file | file_ext | file_dir | input_order | shuffled_order | participant_id | stimulus | timepoint |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S001 | P1_picnic_post | .cha | input/chat | 1 |  | P1 | picnic | post |
| S002 | P1_picnic_pre | .cha | input/chat | 2 |  | P1 | picnic | pre |
| S003 | P2_picnic_pre | .cha | input/chat | 3 |  | P2 | picnic | pre |

### Sheet: utterances

| sample_id | utterance_id | position | position_sub | speaker | utterance | comment |
| --- | --- | --- | --- | --- | --- | --- |
| S001 | U0001 | 1 | 0 | INV | Please tell the picnic story again. |  |
| S001 | U0002 | 2 | 0 | PAR | The family brought food to the park. |  |
| S001 | U0003 | 3 | 0 | PAR | The little girl [/] the little girl pours juice. |  |
| S001 | U0004 | 4 | 0 | PAR | Then they share sandwiches. |  |
| S001 | U0005 | 5 | 0 | INV | Anything else? |  |
| S001 | U0006 | 6 | 0 | PAR | Yes, the dog waits beside them. |  |
| S001 | U0007 | 7 | 0 | PAR | The day is quiet. |  |
| S002 | U0001 | 1 | 0 | INV | Tell me what is happening in the picnic picture. |  |

## Notes

These files are fully synthetic and regenerated from packaged YAML specs. The markdown preview shows only selected rows and snippets; the generated workbook contains the complete example output.
