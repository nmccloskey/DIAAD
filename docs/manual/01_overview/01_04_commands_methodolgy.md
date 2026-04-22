# Commands by Module

## Purpose

DIAAD's commands are best understood as modules within a broader research workflow rather than as isolated technical utilities. Each module corresponds to a recurring methodological task in discourse analysis: preparing transcripts, creating coding materials, evaluating reliability, summarizing coded data, or deriving rate-based and discourse-level outcomes.

This section therefore describes each module in terms of its role in an analytic pipeline. Technical syntax, file naming conventions, and step-by-step operating instructions are documented elsewhere. Here, the emphasis is on what each module contributes to a study, what kinds of user-defined settings shape its behavior, and what forms of data move into and out of the workflow.

Across modules, settings are drawn primarily from `project.yaml` and `advanced.yaml`. In methodological terms, the most important settings concern:

- metadata fields used to define units of analysis and grouping variables
- reliability sampling fraction and random seed for reproducible subsampling
- number of coders and any associated assignment or blinding procedures
- stimulus or narrative fields used to organize templates and vocabulary analyses
- exclusion rules and normalization choices for transcript-based comparisons
- filenames or fields for derived analysis inputs such as speaking time, word counts, and target-vocabulary resources

## Transcripts Module

The transcripts module supports the earliest stage of a DIAAD workflow: organizing transcript data into analyzable structures and establishing transcription reliability. Methodologically, it converts CHAT transcripts into stable tabular representations, selects a reproducible subset for secondary transcription, allows fresh sampling when reliability subsets have already been used, and evaluates agreement between original and reliability transcripts. This module therefore provides the foundation on which later coding and analysis modules depend.

| Functionality | Settings | Inputs | Outputs |
|-----|-----|-----|-----|
| Create sample- and utterance-level transcript tables from CHAT files | Metadata fields; sample shuffling; random seed | CHAT transcripts | Transcript table workbook with sample and utterance sheets |
| Randomly select samples for reliability transcription | Metadata fields; reliability fraction | Transcript tables when available, otherwise CHAT transcripts | Reliability sample table plus blank reliability CHAT templates |
| Draw a new reliability subset that excludes samples used previously | Reliability fraction | Prior transcription reliability selection workbooks | Reselected reliability sample workbook |
| Evaluate agreement between original and reliability transcripts | Metadata fields; excluded speakers; CLAN stripping; correction preference; lowercase normalization; reliability tag/directory conventions | Original and reliability CHAT transcripts | Reliability evaluation table, alignment files, and summary report |

Methodological note: transcript tables are the shared intermediary that allow later modules to work with consistent sample identifiers, utterance identifiers, and metadata labels even when the original source materials are CHAT files.

## Templates Module

The templates module supports studies that require manual coding outside DIAAD's specialized CU, word-count, or POWERS workflows. Methodologically, these commands prepare blank coding instruments that preserve DIAAD's sample structure while assigning coders, optionally blinding identifiers, and selecting reliability subsets. They are useful when a project involves custom coding schemes that still need the same disciplined sampling and coder-management logic as the built-in modules.

| Functionality | Settings | Inputs | Outputs |
|-----|-----|-----|-----|
| Create utterance-level coding templates for custom coding tasks | Reliability fraction; number of coders; stimulus field; blinding settings; random seed | Transcript tables | Primary utterance template, reliability utterance template, and optional blind codebook |
| Create sample-level coding templates, including balanced bins for sample-wise coding schemes | Reliability fraction; number of bins; number of coders; stimulus field; blinding settings; random seed | Transcript tables | Primary sample template, reliability sample template, and optional blind codebook |

Methodological note: the utterance templates preserve utterance granularity, whereas the sample templates are intended for coding constructs that operate on the sample as a whole or within predefined bins.

## CUs Module

The complete utterance (CU) module supports a full manual coding workflow for identifying propositional content at the utterance level. Methodologically, it prepares coder-facing workbooks, manages reliability subsampling, quantifies coder agreement, derives sample-level summaries from utterance coding, and converts counts into speaking-time-adjusted rates. This module is especially important when the study design requires comparison across coding paradigms, including parallel paradigms such as dialect-sensitive CU systems.

| Functionality | Settings | Inputs | Outputs |
|-----|-----|-----|-----|
| Create CU coding files and reliability subsets for coder assignment | Metadata fields; reliability fraction; number of coders; CU paradigms; excluded speakers; stimulus field; blinding settings | Transcript tables | CU coding workbook, CU reliability workbook, and optional blind codebook |
| Draw a new CU reliability subset that excludes already used samples | Metadata fields; reliability fraction; random seed | Existing CU coding and reliability workbooks | Reselected CU reliability workbook |
| Evaluate CU reliability across primary and reliability coding | CU paradigms | CU coding workbook plus CU reliability workbook | Utterance-level reliability file, sample-level reliability summary, and reliability report |
| Analyze finalized CU coding into utterance- and sample-level summaries | CU paradigms; blinding settings for unblinding analyzed outputs | CU coding workbook | CU analysis by utterance, long sample summary, and wide sample summary |
| Convert CU, SV, and REL totals to per-minute rates | CU sample-summary filename; speaking-time filename; speaking-time field | CU sample summary plus speaking-time table | CU rates table |

Methodological note: CU analysis separates coding from aggregation. Coding decisions are made at the utterance level, but most research interpretations are made from sample-level summaries or speaking-time-adjusted rates.

## Words Module

The words module supports manual or semi-automated word-count workflows. Methodologically, it can generate first-pass word-count workbooks from transcript tables or CU outputs, distribute primary and reliability assignments, evaluate agreement between original and reliability counts, summarize counts at the sample level, and derive speaking-time-adjusted rate measures. It is useful when total verbal output is a primary dependent measure or a complementary outcome alongside CU coding.

| Functionality | Settings | Inputs | Outputs |
|-----|-----|-----|-----|
| Create word-count coding files with automated first-pass counts and reliability subsets | Reliability fraction; number of coders; excluded speakers; blinding settings | Preferably CU coding by utterance, otherwise transcript tables | Word-count coding workbook, word-count reliability workbook, and optional blind codebook |
| Draw a new word-count reliability subset that excludes already used samples | Metadata fields; reliability fraction; random seed | Existing word-count coding and reliability workbooks | Reselected word-count reliability workbook |
| Evaluate reliability of word counts between primary and reliability coding | No additional module-specific settings beyond available files | Word-count coding workbook plus word-count reliability workbook | Reliability results table and reliability report |
| Summarize finalized word counts at the utterance and sample levels | Word-count workbook filename; word-count field; blinding settings for unblinding analyzed outputs | Word-count coding workbook | Cleaned utterance-level word-count table and sample-level word-count summary |
| Convert sample-level word counts to per-minute rates | Word-count sample-summary filename; speaking-time filename; speaking-time field | Word-count sample summary plus speaking-time table | Word-count rates table |

Methodological note: when CU results are available, DIAAD uses them to distinguish utterances that should not contribute to word-count totals, making the word-count workflow more analytically selective than a transcript-only count.

## Powers Module

The POWERS module supports discourse-pragmatic coding using the POWERS framework. Methodologically, it builds coder-facing workbooks from transcript tables, can automate portions of the coding setup, creates reliability subsets, evaluates agreement across continuous and categorical POWERS dimensions, supports reselection of unused reliability samples, and aggregates coded outputs to turn, speaker, and dialog levels. This module is designed for projects that require richer discourse-process measures than lexical or utterance-count indices alone.

| Functionality | Settings | Inputs | Outputs |
|-----|-----|-----|-----|
| Create POWERS coding files and reliability subsets | Metadata fields; reliability fraction; number of coders; excluded speakers; automation setting; blinding settings | Transcript tables | POWERS coding workbook with utterance and section-level sheets, plus POWERS reliability workbook |
| Analyze finalized POWERS coding across discourse levels | No additional module-specific settings beyond available coding files | POWERS coding workbook(s) | POWERS analysis workbook(s) with utterance-, turn-, speaker-, and dialog-level summaries |
| Evaluate reliability for POWERS coding variables | No additional module-specific settings beyond available files | POWERS coding workbook plus POWERS reliability workbook | Reliability results workbook and reliability report |
| Draw a new POWERS reliability subset that excludes already used samples | Metadata fields; reliability fraction; random seed; automation setting | Existing POWERS coding and reliability workbooks | Reselected POWERS reliability workbook |

Methodological note: POWERS differs from the CU and word-count modules in that its analytic outputs are explicitly multi-level, supporting interpretation at the utterance, turn, speaker, and interactional levels.

## Vocab Module

The target vocabulary module supports lexicon-coverage analysis for structured discourse tasks such as narrative elicitation. Methodologically, it estimates how fully a speaker covers the expected lexical content of a stimulus, how efficiently those target forms are produced relative to speaking time, and where the sample falls relative to available normative reference data. This module is especially useful when the research question concerns narrative informativeness or access to target lexical items rather than only total output.

| Functionality | Settings | Inputs | Outputs |
|-----|-----|-----|-----|
| Analyze target vocabulary coverage and efficiency for each sample | Metadata fields; excluded speakers; stimulus field; target-vocabulary resource path | Transcript-derived utterance data with narrative/stimulus labels and speaking time | Workbook with sample-level vocabulary coverage summary and item-level detail table |

Methodological note: this module assumes that the stimulus or narrative label is analytically meaningful, because coverage is interpreted relative to a narrative-specific target lexicon rather than to general vocabulary production.

## Turns Module

The turns module supports analysis of digitally coded conversation-turn sequences. Methodologically, it transforms turn-sequence strings into interactional summaries that characterize participation structure, speaker distribution, marked turn events, session-level dynamics, and transition patterns among speakers. It is intended for studies of conversational organization rather than transcript-based lexical or utterance coding.

| Functionality | Settings | Inputs | Outputs |
|-----|-----|-----|-----|
| Analyze conversation-turn files across speaker, group, session, bin, and transition levels | No additional module-specific settings beyond available turn files | Conversation-turn spreadsheets containing grouped turn strings, and optionally session/bin variables | Analysis workbook with speaker-, group-, bin-, session-, participation-, and transition-level summaries |

Methodological note: unlike most DIAAD modules, this workflow does not begin from CHAT transcripts or transcript tables; it assumes that turn sequences have already been coded into a separate tabular representation.

## Workflow Perspective

Taken together, the modules support a methodological progression that many discourse-analysis projects share:

1. Structure transcripts and establish transcription reliability.
2. Generate coder-facing workbooks or templates for the relevant unit of analysis.
3. Evaluate reliability for manual coding tasks.
4. Summarize coded outputs at the utterance, sample, turn, speaker, or dialog level as appropriate.
5. Derive rate-based or lexicon-based outcome measures when the research question calls for normalization or stimulus-referenced interpretation.

In practice, not every study will use every module. DIAAD is designed so that investigators can assemble only the portions of the workflow needed for a given design while retaining a consistent logic for metadata handling, reproducible sampling, coder management, and research-ready outputs.
