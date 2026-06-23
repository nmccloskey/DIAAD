# Coder Identifier Columns Research Context

Coder identifier columns support project logistics. They help a research team distribute coding labor, separate primary and reliability assignments, and audit which rows were intended for which coder.

They are not evidence about the language sample itself.

## Workflow Transparency

Manual coding projects often need two kinds of structure:

- stable record identifiers, such as sample and utterance IDs;
- assignment identifiers, such as coder IDs.

Record identifiers support joins, analysis, and reproducibility. Coder identifiers support human workflow management. Keeping these roles separate helps prevent a project from accidentally treating a coding assignment label as a participant, sample, or analytic grouping variable.

## Reliability Design

Coder identifiers can make reliability workflows easier to administer because they show who should code a selected sample. They are especially helpful when a project distributes primary coding across multiple coders and then routes reliability samples to an alternate coder.

The presence of a `coder_id` column does not by itself guarantee independent coding. A project still needs a protocol for coder training, masking, adjudication, and handling automatically prefilled values.

## Analysis Boundary

DIAAD analysis commands intentionally do not require coder identifier columns. By the time analysis runs, the relevant question is usually whether the completed coding fields are valid for the project's scoring protocol.

This boundary is useful for practical reasons too. Teams sometimes remove administrative columns before analysis, combine coding files from outside DIAAD, or receive completed workbooks where coder assignments were tracked separately. Those files can still be analyzable when the required substantive columns are present.

## Reporting

Methods reporting may mention coder assignment procedures, number of coders, reliability subset size, and independence procedures. It usually should not describe `coder_id` itself as a measured variable unless the project explicitly analyzes coder effects outside DIAAD.

If a project does analyze coder effects, do that in a separate statistical workflow using an intentionally prepared dataset. DIAAD's built-in analysis outputs treat coder IDs as administrative metadata.

## Read Next

- Reliability research context: `docs/manual/05_functionalities/12_reliability_selection_evaluation_reselection/03_research_context.md`
- Revision handling research context: `docs/manual/05_functionalities/11_revision_handling/03_research_context.md`
- Coder identifier implementation notes: `docs/manual/05_functionalities/18_coder_identifier_columns/04_implementation_notes.md`
