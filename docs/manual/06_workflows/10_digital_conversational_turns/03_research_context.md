# Digital Conversational Turns Research Context

Digital Conversational Turns is a compact coding paradigm for conversational sequence. It is intended to reveal patterns in participation and speaker-to-speaker transitions, not lexical content or transcript-level error patterns.

## Why DCT Can Be Useful

A simple tally can show how many turns each speaker took. A DCT sequence preserves order. That order can support transition matrices, participant-to-participant versus clinician-to-participant ratios, session or bin summaries, and sequence-similarity reliability metrics.

This can make DCT analytically generative without requiring full utterance transcription.

## Relation To Transcription

Manual DCT can be transcriptionless or pre-transcription. It may be useful when full transcription is too costly, when conversational dynamics are the main question, or when a project wants an early measure before transcript completion.

If a full transcript already exists, manual DCT coding may be redundant because speaker sequence information is already present in transcript-table speaker tags. In that situation, DIAAD's transcript-table fallback can summarize speaker dynamics directly from the transcript table.

## Relation To POWERS

DCT and POWERS should be kept conceptually separate. DCT focuses on turn sequence and participation. POWERS is a transcript-based coding paradigm for utterance-level dialog variables and selected linguistic measures.

For clinician-client dyads, POWERS may be the more direct transcript-based workflow. DCT becomes especially interesting when there are multiple client or participant speakers and the project cares about direct interactions among them.

## Coding Assumptions

The current manual DCT parser assumes clear linear sequences and single-digit speaker codes. This makes coding efficient, but it creates limitations:

- more than ten speaker categories cannot be represented with the current digit-only convention;
- overlapping talk may not fit a simple sequence;
- ambiguous turn boundaries can reduce reliability;
- dot-marker rules need to be defined in the project protocol.

## Reliability

Reliability evaluation compares both counts and sequence similarity. Low agreement can reflect disagreement about speaker identity, turn boundaries, sequence order, dot markers, or workbook row structure.

## Draft Review Notes

Before publication, review DCT's transcriptionless/pre-transcription framing, overlap limitations, and the relation between manual DCT workbooks and transcript-table-derived speaker dynamics.

## Read Next

- Digital Conversational Turns module research context: `docs/manual/04_modules/07_digital_conversational_turns/03_research_context.md`
- `turns evaluate` research context: `docs/manual/04_modules/07_digital_conversational_turns/05_commands/02_evaluate/03_research_context.md`
- `turns analyze` research context: `docs/manual/04_modules/07_digital_conversational_turns/05_commands/04_analyze/03_research_context.md`
