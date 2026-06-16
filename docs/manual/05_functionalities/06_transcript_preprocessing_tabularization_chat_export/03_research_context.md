# Transcript Preprocessing, Tabularization, Auto-Tabularization, and CHAT Export Research Context

Transcript tabularization is not only a file-format conversion. In DIAAD, it creates the shared data structure that lets discourse analysis move between transcript text, sample metadata, manual coding, reliability checks, blinding, and quantitative summaries.

## Why The Table Matters

Discourse workflows often have many dependent artifacts. A single transcript may lead to utterance-level coding files, sample-level coding files, reliability subsets, rate denominators, blind codebooks, and analysis summaries. If these artifacts do not share stable identifiers, later joins become fragile and difficult to audit.

DIAAD's transcript tables solve this by separating:

- sample-level information, such as source file context and metadata;
- utterance-level information, such as speaker, text, comment, and position;
- diagnostics that should be reviewed before downstream analysis.

This is similar in spirit to a small relational data model, but it stays in an Excel workbook because manual review and hand coding remain central to many DIAAD workflows.

## Why Explicit Tabularization Is Preferred

The default `auto_tabularize: false` setting is methodologically conservative. It nudges users to create transcript tables deliberately, inspect them, and carry a reviewed version forward.

That matters because a transcript table is not neutral bookkeeping. Its sample identifiers, utterance identifiers, metadata fields, and speaker labels shape later coding and analysis. Recreating the table later from a different input folder, shuffle setting, or file set can create a table that looks plausible but no longer corresponds to earlier coding work.

## CHAT Export As A Derived Artifact

CHAT export supports revision workflows, but the exported files should be understood as derived from the reviewed table. This is useful when a project needs revised `.cha` files, but it is not a claim that the original CHAT file can always be reconstructed exactly.

For research reporting, the table usually remains the more important artifact: it records the identifiers, metadata, utterance rows, and revision state that DIAAD uses downstream.

## Read Next

- Functional overview: `docs/manual/01_overview/04_functional_overview.md`
- Transcript tabularization feature: `docs/manual/03_features/01_transcript_tabularization.md`
- Revision handling research context: `docs/manual/05_functionalities/11_revision_handling/03_research_context.md`
