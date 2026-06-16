# Digital Conversational Turns Research Context

Digital Conversational Turns is a coding approach for representing conversational turn sequences as compact digit strings. It is intended for questions about participation, turn distribution, and conversational sequence rather than lexical content.

## Beyond Turn Tallies

A simple turn tally records how many turns each speaker took. DCT preserves more structure by recording the order of turns. A sequence such as `0.1..23.0.12` can be parsed into speaker counts, marker counts, session or bin summaries, transition matrices, and sequence-similarity measures for reliability.

That sequence structure can support analyses of who participates, how participation changes across bins or sessions, and which speaker-to-speaker transitions are common. For example, a project can distinguish participant-to-participant transitions from clinician-to-participant transitions, which may be useful when conversation-focused interventions aim to increase direct participant interaction.

## Transcriptionless Or Pre-Transcription Use

DCT is most useful when a project needs a lower-burden measure before, instead of, or alongside full transcription. If a complete transcript already exists, turn order may already be available through speaker tiers, and separate manual DCT coding may be redundant.

The current DIAAD command surface exposes `turns evaluate` and `turns analyze`, not a DCT file-generation command. Manual DCT workbooks are prepared outside the current `turns` command surface. When no primary DCT workbook is found, `turns analyze` can instead use ordered transcript-table speaker tags as a transcript-table-informed speaker-dynamics summary.

## Speaker Codes And Sequence Assumptions

The current DCT parser treats each digit as one speaker code. In the intended convention, `0` represents the clinician or other non-client interlocutor category, and digits `1` through `9` represent client or participant speakers.

This design is compact and easy to code with a number pad, but it assumes no more than ten speaker categories. It also assumes that turn order can be represented as a clear linear sequence. Overlap, simultaneous talk, or group conversations with many speakers can make that assumption less realistic.

Future protocols could represent overlapping turns with additional syntax, but that would make coding and reliability interpretation more complex. The current DIAAD implementation should therefore be used most cautiously for settings where overlapping turns are frequent or analytically central.

## Reliability

Reliability evaluation uses both count and sequence information. Count summaries ask whether coders attributed similar numbers of turns to each speaker within each sample, session, and bin. Sequence summaries ask whether the two turn strings are similar in order, using Levenshtein distance and similarity.

Low agreement can reflect disagreement about turn boundaries, speaker assignment, digit-string syntax, or how to handle overlapping or ambiguous turns. Reliability results should be interpreted alongside the project's DCT coding protocol and any training or adjudication process.

## Read Next

- Digital Conversational Turns quickstart: `docs/manual/04_modules/07_digital_conversational_turns/01_quickstart.md`
- Configuration: `docs/manual/02_operation/04_configuration.md`
- Generated Example I/O: `docs/manual/03_features/04_generated_example_io.md`
