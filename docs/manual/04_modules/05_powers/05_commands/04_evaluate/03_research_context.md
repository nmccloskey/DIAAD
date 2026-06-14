# `powers evaluate` Research Context

POWERS reliability evaluation summarizes whether two coding passes applied the POWERS coding fields similarly enough for the project's purpose. DIAAD reports multiple kinds of evidence because POWERS includes both count-like and categorical fields.

## Continuous And Categorical Evidence

For count-like fields, DIAAD reports absolute differences, percent differences, percent similarity, exact agreement, within-one-count agreement, ICC(2,1), missingness, and variance diagnostics.

For categorical fields, DIAAD reports percent agreement and Cohen's kappa. `turn_type` is treated as a categorical code. `collab_repair` is summarized as presence or absence of a repair value.

These summaries answer different questions. A count metric can have high within-one agreement but low ICC if most utterances have the same value or if nonzero values are sparse. A categorical metric can have high percent agreement but unstable kappa when one code dominates.

## Automation And Reliability

Some POWERS fields can be populated by `powers files` automation before human review. If both primary and reliability coders leave the same automated values unchanged, reliability for those fields partly reflects shared first-pass automation rather than independent human agreement.

For that reason, projects should decide how automated fields are reviewed, corrected, and interpreted before treating reliability metrics as evidence of coder consistency.

## Section E Boundary

Section E fields are not part of DIAAD's current POWERS reliability evaluation. They are better understood as sample-level note or descriptor fields unless a project defines a separate coding and analysis plan for them.

## Interpretation Cautions

The reliability report is a diagnostic aid, not a decision rule. Interpret it alongside the POWERS coding manual, coder training history, the distribution of observed values, and any planned adjudication or recoding process.
