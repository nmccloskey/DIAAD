# Digital Conversational Turns Module Quickstart

The Digital Conversational Turns module supports turn-sequence coding when a project wants to represent conversational turns directly. After definition, this manual may refer to the module as DCT.

DCT coding can be useful when a project wants a lower-burden alternative or precursor to full transcript analysis. Manual DCT coding uses compact turn-string workbooks rather than utterance-level transcript text.

## Commands

| Command | Main use |
|---|---|
| `diaad turns evaluate` | Evaluate DCT reliability. |
| `diaad turns analyze` | Analyze completed DCT turn strings. |

## Typical Sequence

```text
manual turn-string coding
turns evaluate
turns analyze
```

Coders enter a turn string such as:

```text
0.1..23.0.12
```

Digits identify speakers. By convention in the bundled guidance, `0` is the clinician or other non-client interlocutor category, and digits `1` through `9` identify client or participant speakers. Dots are optional markers that DIAAD preserves and summarizes.

The current parser treats each digit as one speaker code, so it does not support participant identifiers of `10` or above as multi-digit IDs.

## Common Outputs

| Step | Typical outputs |
|---|---|
| Reliability evaluation | `turns_reliability/conversation_turns_reliability_results.xlsx`, report, alignments |
| Analysis | `conversation_turns_analysis.xlsx` or similarly named analysis workbook with speaker, group, session, bin, participation, and transition summaries where available |

## Read Next

- Transcript tabularization: `docs/manual/03_features/01_transcript_tabularization.md`
- Configuration: `docs/manual/02_operation/04_configuration.md`
- Digital Conversational Turns research context: `docs/manual/04_modules/07_digital_conversational_turns/03_research_context.md`

Later command pages describe turn-string syntax, bin/session columns, reliability comparison, and transition outputs.
