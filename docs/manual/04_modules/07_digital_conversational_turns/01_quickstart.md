# Digital Conversational Turns Module Quickstart

The Digital Conversational Turns module supports turn-sequence coding when a project wants to represent conversational turns directly, often before or instead of full transcription. After definition, this manual may refer to the module as DCT.

## Commands

| Command | Main use |
|---|---|
| `diaad turns files` | Create DCT coding and reliability workbooks. |
| `diaad turns evaluate` | Evaluate DCT reliability. |
| `diaad turns reselect` | Select replacement DCT reliability material. |
| `diaad turns analyze` | Analyze completed DCT turn strings. |

## Typical Sequence

```text
turns files
manual turn-string coding
turns evaluate
turns analyze
```

When transcript tables are available, `turns files` can use them to scaffold sample and bin rows. The analytic value of DCT is strongest when turn order is not already fully represented in transcript content, or when a project deliberately codes turns before transcription.

## Common Outputs

| Step | Typical outputs |
|---|---|
| File generation | `coding_templates/conversation_turns_template.xlsx`, reliability template, codebook |
| Reliability evaluation | `turns_reliability/conversation_turns_reliability_results.xlsx`, report, alignments |
| Reselection | `reselected_turns_reliability/` |
| Analysis | A DCT analysis workbook with speaker, group, session, bin, participation, and transition summaries where available |

## Read Next

- Transcript tabularization: `docs/manual/03_features/01_transcript_tabularization.md`
- Configuration: `docs/manual/02_operation/04_configuration.md`

Later command pages describe turn-string syntax, bin/session columns, reliability comparison, and transition outputs.
