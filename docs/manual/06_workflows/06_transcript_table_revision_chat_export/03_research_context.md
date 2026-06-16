# Transcript Table Revision and CHAT Export Research Context

Transcript revision is normal in discourse-analysis workflows. Errors often become visible only after transcripts are tabularized, coded, compared, or prepared for sharing. DIAAD therefore treats transcript tables as revisable canonical data, not as disposable intermediate files.

## Flexible But Stable

The transcript table design tries to balance two needs:

- stable identifiers and database-like joins for computation;
- human-editable spreadsheets for review and correction.

This balance is important because discourse analysis still depends on human judgment. A fully opaque database representation may be harder for users to inspect, while loose transcript files may be harder to join reliably with coding, metadata, and analysis outputs.

## Revision Risk

Revision is powerful because it keeps the table current. It is risky because downstream files may already depend on older rows.

Changing an utterance can change word counts, target-vocabulary coverage, complete utterance judgments, POWERS coding, rates, and reliability comparisons. Changing identifiers can break joins entirely.

For that reason, revision should be treated as a workflow event. Archive old outputs, document what changed, and rerun or recode affected artifacts.

## CHAT Export

CHAT export is useful when revised table content needs to leave DIAAD's table-centered workflow. It supports sharing, CLAN analysis, and external tools that expect `.cha` files.

The exported files are derived artifacts. The transcript table remains the internal DIAAD source of truth unless the project intentionally chooses another representation.

## Read Next

- Transcript tabularization research context: `docs/manual/04_modules/01_transcripts/05_commands/01_tabularize/03_research_context.md`
- Revision handling research context: `docs/manual/05_functionalities/11_revision_handling/03_research_context.md`
- Exact file name matching: `docs/manual/03_features/03_exact_file_name_matching.md`
