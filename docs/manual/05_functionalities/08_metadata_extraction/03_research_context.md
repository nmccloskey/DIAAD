# Metadata Extraction Research Context

Metadata extraction is part of reproducible data preparation. It makes project context visible in the same workbook that carries transcript identifiers and utterance rows.

## Why Metadata Belongs Near The Transcript

Discourse data often depend on sample-level context: task, session, condition, participant group, site, or stimulus. If those values remain only in file names or external spreadsheets, later analysis can become hard to audit. Writing them into the transcript table makes the assumptions explicit at the point where transcript-derived data first enter DIAAD.

## Metadata Is Not Ground Truth

Automated extraction should be treated as a first pass. A successful extraction means that DIAAD found a configured pattern or value. It does not prove that the file was named correctly, the folder was organized correctly, or the metadata label is analytically valid.

The `metadata_mismatch` column and `metadata_mismatches` sheet are designed to keep uncertainty visible. Review these diagnostics before using the table for coding, blinding, grouping, or statistical export.

## Reproducibility Value

When metadata is captured in the transcript table, later analysts can reconstruct how a result was grouped or joined. This is especially helpful when manually coded files, blind codebooks, and analysis exports circulate separately from the original transcript folder.

## Read Next

- Run provenance and audit artifacts: `docs/manual/05_functionalities/03_run_provenance_audit_artifacts/03_research_context.md`
- Transcript preprocessing research context: `docs/manual/05_functionalities/06_transcript_preprocessing_tabularization_chat_export/03_research_context.md`
- Blinding research context: `docs/manual/05_functionalities/14_blinding_unblinding_auto_blind/03_research_context.md`
