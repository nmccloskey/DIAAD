# Monologic Narrative Target Vocabulary Coverage Research Context

Target Vocabulary Coverage is a resource-driven analysis. The resource defines the construct that is being measured: which target words exist, which variants count, and which norms may be used.

## CoreLex-Style Built-Ins

DIAAD includes built-in CoreLex-style resources for five narrative tasks. These provide a convenience layer for canonical narrative analyses and make it possible to run batch, database-friendly coverage workflows.

The built-ins should be reported as built-in CoreLex-style resources, not as proof that every possible TVC resource has the same validation status.

## Custom Resources

Custom TVC resources generalize the CoreLex-like paradigm. They can support task-specific prompts, treatment vocabularies, procedural descriptions, or other lexicons.

This flexibility is useful, but it moves validation responsibility to the project. A custom resource can pass DIAAD's structural checker while still needing independent evidence that the target words and accepted variants support the intended interpretation.

## Coverage Versus Quantity

TVC is not a total word-count measure. It asks how much of a predefined lexicon appeared. Measures such as `lexicon_coverage`, `num_base_forms_produced`, and `num_core_token_matches` should therefore be interpreted as target-vocabulary measures rather than general language quantity.

## Norms

When resources declare norm tables, DIAAD can retrieve those tables and compute percentile fields locally. Norm retrieval may require network access. Percentiles should be interpreted only when the comparison group and task fit the project.

## Draft Review Notes

Before publication, review CoreLex-related citation language, built-in validation claims, norm-source wording, and any custom-resource validity cautions.

## Read Next

- Target Vocabulary Coverage research context: `docs/manual/04_modules/06_target_vocabulary_coverage/03_research_context.md`
- Target Vocabulary Resource Management research context: `docs/manual/05_functionalities/16_target_vocabulary_resource_management/03_research_context.md`
- Word Counting research context: `docs/manual/06_workflows/08_monologic_narrative_word_counting/03_research_context.md`
