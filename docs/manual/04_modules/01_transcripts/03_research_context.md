# Transcripts Research Context

Transcript tables are DIAAD's main bridge between qualitative transcript files and database-like analysis workflows. The module starts from CHAT transcripts, but the broader design principle is more general: stable identifiers and organized tables make it possible to connect transcript content, sample metadata, coding files, reliability subsets, blinding codebooks, and analysis summaries.

## Transcript Tables As Shared Structure

DIAAD implements this structure with Excel workbooks rather than a database server. The `samples` sheet stores one row per transcript/sample, while the `utterances` sheet stores one row per utterance. The default identifiers, `sample_id` and `utterance_id`, let downstream files refer back to the same analytic units.

This matters for discourse analysis because projects rarely move in a straight line from transcript to final statistic. Researchers may revise transcripts, select reliability subsets, blind coding workbooks, compare primary and reliability coding, add speaking times, and aggregate across metadata groups. Transcript tables give those steps a common reference system.

## Reliability And Revision

The transcript module also supports transcription reliability. Selection and reselection commands help allocate samples for independent transcription review. Evaluation compares original and reliability transcripts, including text-level differences that may reflect words, nonwords, fillers, spelling variation, utterance segmentation, and other transcript decisions.

The module's revision path is deliberately table-centered. If errors are found during coding, users can correct transcript-derived tables while preserving identifiers. The `position` and `position_sub` columns make ordering explicit, including inserted rows. `transcripts chats` can then export CHAT-style files from revised tables, but this should be understood as a revision-export path rather than a guarantee that every original formatting detail can be perfectly reconstructed.

## Scope

Transcript tabularization is the usual first step for transcript-based DIAAD workflows, but it is not universal. Transcriptionless or pre-transcription workflows, externally prepared tables, and existing DIAAD transcript tables may enter the system differently.

The important research practice is to keep the analytic units stable and auditable. For CHAT-based discourse data, the Transcripts module is DIAAD's standard way to do that.

## Draft Review Notes

Before publication, review the reliability-method wording, especially any later thresholds or character-alignment interpretations added to command-level pages. Also review the CHAT export language against the intended revision workflow.
