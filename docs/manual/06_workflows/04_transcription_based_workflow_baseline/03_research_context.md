# Transcription-Based Workflow Baseline Research Context

Transcript-based discourse analysis depends on transcript quality, stable identifiers, and a clear record of how text moved from raw recording to coded or analyzed data. The baseline workflow helps organize those dependencies.

## Why A Baseline Matters

Transcript work is often distributed across people and stages. One person may prepare an initial transcript, another may review it, another may complete reliability transcription, and another may code utterances. Without a shared data representation, later analysis can become a collection of loosely related files.

DIAAD's transcript table is meant to be the shared representation. It gives each sample and utterance a stable coordinate so metadata, coding decisions, reliability rows, blinding codebooks, and analysis outputs can be joined and audited.

## Reliability Before Analysis

Transcription reliability is one early check on whether the transcript set is suitable for downstream discourse measures. DIAAD reports character-level and alignment-based evidence, but the project still needs a transcription protocol and a review policy.

The baseline places reliability before downstream analysis because transcript errors can propagate into complete utterance coding, word counts, target-vocabulary matching, POWERS coding, and rates.

## Tables As A Human-Editable Relational Model

The transcript table design is intentionally different from hiding the data inside a database server. It gives users an Excel workbook they can inspect and edit, while preserving enough database logic for reproducible joins and downstream file generation.

That flexibility comes with responsibility. Changes to utterance text, row identity, speaker labels, or metadata after downstream coding can invalidate derived files. The revision workflow explains how to edit cautiously.

## Exceptions

Not every DIAAD use case begins with CHAT tabularization. A project may start from existing transcript tables, external tables with stable identifiers, generic templates, or Digital Conversational Turns. Still, for transcript-based discourse analysis, the transcript table is DIAAD's primary scaffold.

## Read Next

- Transcript revision and CHAT export research context: `docs/manual/06_workflows/06_transcript_table_revision_chat_export/03_research_context.md`
- Transcription reliability research context: `docs/manual/06_workflows/05_transcription_reliability/03_research_context.md`
- Revision handling: `docs/manual/05_functionalities/11_revision_handling/03_research_context.md`
