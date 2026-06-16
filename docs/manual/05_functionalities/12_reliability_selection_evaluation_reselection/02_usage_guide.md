# Reliability Selection, Evaluation, and Reselection Usage Guide

Reliability behavior varies by module because different discourse measures have different units and scoring rules. This page explains the shared workflow logic; module command pages explain exact metrics and file layouts.

## Selection

Reliability selection chooses a subset of samples for independent review.

For transcription reliability, `transcripts select` prefers an existing transcript table when one is available. If no transcript table is found and CHAT files are available, it can fall back to the CHAT file set. The output records both selected samples and the full sample frame.

For coding modules such as Complete Utterances, Word Counting, and POWERS, reliability material is usually created by the module's file-generation command. If a sample is selected for reliability, the relevant rows for that sample are included in the reliability workbook.

The target count is based on `project.reliability_fraction`. The shared subset-size helper calculates the ceiling of `fraction * number_of_samples`, with a minimum of one sample.

## Evaluation

Run evaluation after primary and reliability files have both been completed and reviewed.

The evaluation command merges primary and reliability data using the configured identifiers, then writes a report and a detailed workbook. Metrics differ by module:

| Area | Main comparison style |
|---|---|
| Transcription | Character-level transcript comparison, including Levenshtein and Needleman-Wunsch metrics. |
| Complete Utterances | Utterance-level categorical agreement and kappa, plus sample-level total reliability summaries. |
| Word Counting | Paired utterance count differences, percent similarity, agreement flags, and ICC summaries. |
| POWERS | Continuous score differences and ICC where applicable; categorical percent agreement and kappa. |
| Digital Conversational Turns | Turn-count agreement, sequence-similarity metrics, and alignment outputs. |

These summaries help identify disagreement. They do not decide whether a project protocol is valid or whether either coder applied the protocol correctly.

## Reselection

Reselection is a fallback, not a routine required step.

Use reselection when:

- the first reliability round does not meet the project's threshold;
- too few reliability samples were usable;
- a coding or transcription problem requires a fresh independent subset;
- additional reliability material is needed after adjudication.

Reselection commands avoid sample IDs that already appear in prior matched reliability files when the source files provide enough information to identify them.

Digital Conversational Turns currently exposes `turns evaluate` and `turns analyze` in the CLI command registry. It does not currently expose a `turns reselect` command, even though reselection helpers exist in source.

## Reading Results

When reliability is low, inspect the detailed workbook before treating the summary as a final verdict. Common causes include:

- ambiguous coding protocol language;
- transcript segmentation differences;
- unreviewed first-pass automation;
- coder training differences;
- mismatched identifiers;
- stale coding files after transcript revisions;
- low score variance that makes ICC difficult to interpret.

## Read Next

- Revision handling: `docs/manual/05_functionalities/11_revision_handling/02_usage_guide.md`
- General sample subsetting: `docs/manual/05_functionalities/13_sample_subsetting_resubsetting/02_usage_guide.md`
- Configuration: `docs/manual/02_operation/04_configuration.md`
- Testing: `docs/manual/02_operation/05_testing.md`
