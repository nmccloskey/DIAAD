# `cus evaluate` Research Context

CU reliability evaluation summarizes how consistently a primary and reliability coding pass applied the project's CU coding scheme. The command reports useful evidence, but it does not determine whether a project's reliability evidence is sufficient for every research purpose.

## What The Command Measures

DIAAD treats SV and REL as the two coded components from which CU is derived. A derived CU value is positive only when both SV and REL are coded `1`. Reliability is then summarized for SV, REL, and derived CU.

The command reports both row-level and sample-level views:

- utterance-level agreement and Cohen's kappa for categorical coding decisions;
- sample-total ICC metrics for count-like totals;
- legacy descriptive agreement summaries, including sample-level 80-percent flags;
- coverage diagnostics showing how much of the primary coding file appears in the reliability comparison.

## Interpretation Cautions

No single metric is enough by itself. Percent agreement is easy to understand but does not account for chance agreement. Kappa can be difficult to interpret when codes are rare or variance is low. ICC metrics summarize agreement in sample-level totals, which may be more relevant for some quantitative summaries but less direct for utterance-level coding decisions.

Use the report to identify the overall pattern, then inspect the utterance-level workbook when metrics are low, undefined, or unexpectedly high. The project should decide in advance how reliability evidence will be reviewed, what counts as an acceptable disagreement, and how recoding or adjudication will be handled.

## Relation To The Coding Protocol

DIAAD can compute agreement on completed CU columns, but it cannot judge whether coders applied a valid CU definition. The reliability output is meaningful only in relation to the project's coding manual, training process, and decision rules for ambiguous utterances.
