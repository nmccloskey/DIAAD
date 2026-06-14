# `transcripts tabularize` Research Context

Transcript tabularization is more than file conversion. In DIAAD, it establishes the shared sample and utterance structure that later coding, reliability, blinding, and analysis steps use to stay aligned.

## Why Tables Matter

Discourse analysis workflows often revisit earlier decisions. A transcript may be corrected after coding begins, a subset may be selected for reliability review, or a project may need parallel coding files grouped by sample, utterance, stimulus, coder, or metadata. A tabular representation gives those later steps stable identifiers rather than requiring each command to reinterpret raw transcript files independently.

DIAAD's default table design separates sample-level information from utterance-level information:

- `samples` stores the transcript/sample as the unit of metadata and selection;
- `utterances` stores the transcript content as ordered utterance rows;
- identifiers link those units across later workbooks.

This is not a claim that the transcript table is the analysis itself. It is an audit-friendly scaffold for moving between transcript text, human coding, reliability procedures, and aggregate outputs.

## Revision And Auditability

Transcript projects rarely move in a perfectly linear way. Table-based revision lets users correct transcript-derived content while preserving the identifiers that downstream coding files depend on.

The main methodological caution is that revisions should be deliberate. Changing an utterance row, inserting a row, or altering a sample-level metadata value can change later analyses. The table format makes those changes visible, but the project team still needs a revision policy.

## Limits Of The Command

`transcripts tabularize` does not validate a study's transcription conventions, sampling plan, coding manual, or metadata ontology. It also does not make later blinding or reliability decisions valid on its own. It creates structured transcript data so those decisions can be made and audited more clearly.

For the broader module framing, see Transcripts research context (`docs/manual/04_modules/01_transcripts/03_research_context.md`).
