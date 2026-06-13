---
object_type: command
object_types:
- command
object_id: transcripts.chats
command_id: transcripts.chats
canonical_command: transcripts chats
module_id: transcripts
title: Transcript CHAT File Export Example
view: example_io
view_label: Example I/O
view_order: 50
slot: examples
source_manual: generated_example_io
generated: true
---

# Transcript CHAT File Export Example

This example demonstrates how `diaad transcripts chats` converts transcript table rows back into CHAT-style transcript files.

## Command

```bash
diaad transcripts chats --config config
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
        chat_files/
          P1_picnic_pre_cha_source_input_chat_P1_picnic_pre_0.cha
          P2_picnic_pre_cha_source_input_chat_P2_picnic_pre_0.cha
          P1_picnic_post_cha_source_input_chat_P1_picnic_post_0.cha
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

## Input Snippet

The command uses `diaad_data/input/transcript_tables/transcript_tables.xlsx`.

## Output Preview

`diaad_data/output/diaad_YYMMDD_HHMM/chat_files/P1_picnic_post_cha_source_input_chat_P1_picnic_post_0.cha`

```text
@Begin
@Languages:	eng
@Participants:	PAR0 Participant, INV Investigator
@ID:	eng|corpus_name|PAR0|||||Participant|||
@ID:	eng|corpus_name|INV|||||Investigator|||
*INV:	Please tell the picnic story again.
*PAR:	The family brought food to the park.
*PAR:	The little girl [/] the little girl pours juice.
*PAR:	Then they share sandwiches.
*INV:	Anything else?
*PAR:	Yes, the dog waits beside them.
*PAR:	The day is quiet.
@End
```

## Notes

The synthetic filenames include the source workbook and sample identifiers so exported CHAT files can be traced back to their table rows.
