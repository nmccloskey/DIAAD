# `transcripts evaluate` Research Context

`transcripts evaluate` provides text-level evidence about transcription agreement. It is useful for auditing transcription reliability, but its metrics should be interpreted as aids to review rather than as a complete theory of transcript quality.

## What The Metrics Capture

The command compares processed utterance text from an original transcript and a reliability transcript. It reports token counts, character counts, edit-distance metrics, and global alignment metrics.

Levenshtein similarity is a normalized character-level similarity score. Higher values mean fewer character-level edits would be needed to transform one processed transcript into the other. The generated text report groups similarity values into broad bands, but those bands should be treated as operational summaries rather than universal standards.

Needleman-Wunsch alignment gives a global sequence-alignment view of the two processed texts. The alignment files are especially useful when a score needs human interpretation, because they show where the transcripts differ.

## What The Metrics Do Not Capture

The metrics do not know which transcript is substantively correct. They also cannot distinguish all methodologically important differences from harmless formatting differences unless the project's processing settings are appropriate.

Examples of decisions that can affect interpretation include:

- whether investigator speech should be excluded;
- how CLAN markup should be stripped;
- whether corrected forms should replace original marked forms;
- whether casing differences should count;
- whether segmentation differences matter for the project.

The command can summarize differences, but project teams still need a transcription protocol and a review policy for resolving disagreements.

## Practical Interpretation

Use the workbook to identify cases that need review, the report to summarize the run, and alignment files to diagnose specific pairs. For low or surprising scores, inspect the original and reliability transcripts rather than relying on the scalar metrics alone.

Before publication, align the project's reporting language with its transcription protocol and any discipline-specific expectations for reliability evidence.
