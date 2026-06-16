# General Sample Subsetting and Re-Subsetting Research Context

Subsetting is a methodological decision. A random subset can make reliability review, piloting, or quality control feasible, but the subset should match the purpose of the review.

## Beyond Reliability

`templates subset` was built with reliability-style use cases in mind, but it is deliberately general. A project might use it to choose material for coder training, pilot coding, adjudication review, or a local protocol that DIAAD does not otherwise understand.

The output is not an analysis result. It is a documented selection artifact.

## Re-Subsetting

Re-subsetting helps avoid reusing already reviewed samples. This is useful when the first round did not provide enough usable material or when a project needs a second independent review set.

The `exclude` column is intentionally simple. It documents which samples should be unavailable for the next draw without asking DIAAD to infer why a sample was excluded.

## Sampling Limits

A random subset is only as meaningful as the frame it is drawn from. If the input workbook omits samples, duplicates samples unintentionally, or mixes incompatible groups, the selection will inherit those problems.

For stratified or blocked sampling, project teams may need to prepare separate input workbooks or use a project-specific selection process outside DIAAD.

## Read Next

- Run provenance and audit artifacts: `docs/manual/05_functionalities/03_run_provenance_audit_artifacts/03_research_context.md`
- Reliability research context: `docs/manual/05_functionalities/12_reliability_selection_evaluation_reselection/03_research_context.md`
- Templates research context: `docs/manual/04_modules/02_templates/03_research_context.md`
