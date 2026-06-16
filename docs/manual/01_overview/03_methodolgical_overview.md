# Methodological Overview

DIAAD is built around a simple premise: discourse analysis becomes more reproducible when transcript content, metadata, coding decisions, reliability subsets, and analysis outputs are organized as connected tables rather than as isolated files. The program uses familiar Excel workbooks for much of this structure, but the underlying logic is relational. Rows represent identifiable units such as samples or utterances, and stable identifiers allow information from different files to be joined, checked, blinded, revised, and summarized.

This page gives a high-level view of DIAAD's methodological role. Later sections document individual modules, commands, and implementation details.

## Why Discourse Workflows Need Infrastructure

Discourse samples are often more clinically and ecologically informative than single-word or sentence-level tasks, because everyday communication usually unfolds across narratives, explanations, conversations, and other extended language contexts (Armstrong et al., 2011; Bryant et al., 2017). That richness also makes discourse data difficult to manage. A single project may involve audio or video recordings, CHAT transcripts, participant metadata, multiple coding paradigms, reliability samples, blinded coder workbooks, speaking-time measurements, and aggregate analysis outputs.

Without a shared data structure, the same sample can appear under different names in different folders, manual coding files can drift away from the transcript version that generated them, and reliability procedures can become difficult to audit. DIAAD addresses this by making the data scaffold explicit. It does not prescribe one research design. Instead, it gives users a structured way to keep the pieces of a design aligned.

## The Transcript Table Scaffold

For most DIAAD workflows, the transcript table workbook is the central artifact. The `transcripts tabularize` command reads CHAT files and writes a workbook with sample-level and utterance-level tables. The sample table stores one row per transcript sample with metadata and a stable sample identifier. The utterance table stores transcript content with utterance identifiers, speaker labels, and links back to the sample table.

This structure supports several methodological needs:

- Metadata can be extracted once and reused across later analyses.
- Manual coding files can be generated from a consistent source.
- Reliability subsets can be sampled from the same population as the primary coding files.
- Blinding can preserve links between transcript content, metadata, and blind codes.
- Revisions can be tracked through stable identifiers and ordering fields.
- External or downstream tables can be joined when they share a stable identifier.

The transcript table is not the only possible DIAAD entry point. Digital Conversational Turns can be coded before transcription or without transcription, and generic templates can support custom coding schemes. Still, DIAAD is primarily transcript centered, and most transcript-based workflows should begin by creating or reusing a canonical transcript table rather than repeatedly regenerating equivalent tables.

For that reason, the `auto_tabularize` setting in the project configuration defaults to `false`. Automatic table creation is convenient, but accidental regeneration can create duplicate canonical representations of the same transcript set and may assign new sample identifiers. The safer default is explicit tabularization followed by deliberate reuse of the resulting workbook.

## Human-in-the-Loop Analysis

DIAAD automates file preparation, table organization, selected first-pass measures, reliability calculations, and output aggregation. It does not replace human methodological judgment.

This is especially important for manual or semi-automated coding paradigms. Complete Utterance coding, word counting, and POWERS coding depend on protocol decisions that should be documented within and across research groups (Dietz & Boyle, 2018; Stark et al., 2020, 2021). DIAAD can prepare coding workbooks and summarize completed codes, but users remain responsible for defining the coding rules that make those codes meaningful.

Word counting illustrates the point. In many settings, word counts can be automated. In clinical discourse samples, however, users may need to omit repetitions, nonword fillers, commentary outside the target task, or minimal prompt responses, depending on the coding protocol (Nicholas & Brookshire, 1993; Forbes et al., 2012). DIAAD can provide first-pass support, but the final analytic value depends on human review.

The same principle applies to POWERS automation. DIAAD can support selected automated first-pass counts using natural language processing, but those outputs should be treated as aids for review, not as replacements for discourse-pragmatic coding judgment.

## Reliability as a Repeated Pattern

Reliability is not a single DIAAD command. It is a repeated pattern across transcription and manual coding workflows:

1. Select a reproducible subset.
2. Complete a second transcription or coding pass.
3. Evaluate agreement.
4. If thresholds are not met, reselect unused material for another reliability round.

For transcription reliability, DIAAD compares original and reliability transcripts at the character level. Character-level methods are useful because they can capture differences across words, nonwords, fillers, spelling variants, and other transcript details. DIAAD reports metrics based on Levenshtein edit distance and Needleman-Wunsch global alignment, including normalized similarity scores and alignment outputs that can help users inspect mismatched transcript regions (Levenshtein, 1966; Needleman & Wunsch, 1970).

For manual coding reliability, the relevant metrics and interpretation depend on the coding paradigm. CU, word counts, POWERS, and DCT each have different output structures and different types of agreement to evaluate. The shared methodological point is that reliability subsets should be sampled reproducibly, evaluated explicitly, and repeated when the original subset does not support adequate confidence.

## Manual Coding and Analysis Modules

DIAAD's analytic modules occupy different places in a discourse-analysis workflow.

Complete Utterances focuses on utterance-level judgments about propositional completeness and related coding decisions. Its outputs are useful when a project needs sample-level summaries of utterance-level manual coding.

Word Counting supports total-output measures while allowing human review of first-pass counts. It is distinct from target vocabulary coverage: word counting asks how much language was produced under a given counting protocol, whereas target vocabulary coverage asks how well a sample covered a stimulus- or task-specific lexicon.

POWERS supports discourse-pragmatic coding for clinician-client or other dialogue data. Unlike measures that summarize only lexical output or utterance counts, POWERS can support summaries across utterances, turns, speakers, and dialogs.

Target Vocabulary Coverage supports CoreLex-style and custom lexicon analyses. Built-in narrative resources provide a convenience layer for select standard stimuli, while custom lexicons can be used for project-specific elicitation tasks. Custom resources should be interpreted cautiously unless their psychometric properties are evaluated for the intended use (Pritchard et al., 2018).

Digital Conversational Turns supports transcriptionless or pre-transcription analysis of turn sequences. It can be useful when a project needs a feasible way to characterize participation patterns, speaker transitions, or conversational dynamics before a full transcript exists. If a transcript is already available, the sequence of turns is already represented in the transcript's speaker tags; a future extension may derive similar profiles directly from transcript tables.

The Templates module is different from these analytic modules. It is supportive infrastructure for coding schemes DIAAD does not directly analyze. It can create utterance-level templates, sample-level templates, speaking-time templates, and general sample subsets while preserving the same logic of identifiers, coder assignment, blinding, and reliability sampling.

## Rates, Coverage, and Normalization

DIAAD separates raw coding or analysis from rate calculation. Users enter speaking time in seconds, usually through a speaking-time template. Rate commands then calculate units per minute. The numerator depends on the module: complete utterance totals, word counts, POWERS variables, or target vocabulary measures.

Rate normalization is optional. It is useful when the research question concerns production relative to time, but not every discourse measure should be interpreted primarily as a rate. Similarly, target vocabulary coverage measures should not be collapsed into ordinary word counts - as the emphasis is not on volume but accuracy with respect to a prespecified lexicon.

## Blinding, De-identification, and Practical Limits

DIAAD supports blinding by replacing configured identifier columns with blind codes and writing codebooks that allow those identifiers to be restored later. This can reduce bias during manual coding and help prepare analysis-ready outputs. It also creates a structured record of how blind identifiers map to canonical sample identifiers.

Blinding should not be confused with full de-identification. Transcript text may contain sensitive details even when names are removed. A file name may be deidentified while the content remains recognizable to someone familiar with the participant or session. A coding team may also be formally blinded but practically unblinded if the coders collected the original data. DIAAD's blinding tools should therefore be understood as one part of a broader privacy and study-design strategy.

For highly sensitive or difficult-to-deidentify data, local CLI use is usually preferable to the hosted web app. The web app is useful and designed around temporary workspaces, but users remain responsible for deciding whether their data are appropriate to upload.

## Configuration, Exact Matching, and Provenance

DIAAD's flexibility comes from configuration. Settings define input and output directories, metadata extraction, reliability fractions, random seeds, excluded speakers, transcript table filenames, coding workbook filenames, identifier columns, target vocabulary resources, blinding behavior, and more.

Some of these settings are methodological, not merely technical. A random seed makes sampling reproducible. A reliability fraction defines the proportion of the dataset staged for evaluation. Excluded speakers determine which transcript rows contribute to counts or analyses. Exact filename matching protects users from accidentally analyzing the wrong workbook when a directory contains multiple similar files.

Each run writes artifacts that help users audit what happened, including effective configuration and other run metadata. These records are especially important when a project has many iterations, when manual coding files are revised, or when outputs need to be traced back to a particular configuration state.

## Using DIAAD Responsibly

DIAAD is best understood as workflow infrastructure. It can make large discourse projects more organized, reproducible, and efficient, but it does not make methodological choices automatically valid. Users should:

- define coding protocols before large-scale annotation;
- check automated first passes before treating them as final data;
- interpret measures in light of their literature-supported psychometric profile;
- preserve canonical transcript tables and identifiers;
- use exact configured filenames deliberately;
- document configuration choices and run artifacts;
- use local processing for data that should not be uploaded;
- treat custom lexicons and custom coding schemes as measures requiring validation.

The later module and command pages provide further details regarding usage, potential research context, and implementation. This overview is meant to supply the broad methodological frame: DIAAD helps users organize and analyze discourse samples in connected, auditable, and database-centered workflows while keeping the human decisions visible.
