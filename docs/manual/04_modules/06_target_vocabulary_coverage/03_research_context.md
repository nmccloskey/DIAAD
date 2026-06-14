# Target Vocabulary Coverage Research Context

Target Vocabulary Coverage measures production of a predefined lexical target set. It is related to CoreLex-style narrative analysis when the resource defines targets for a standard narrative, but DIAAD generalizes the structure so users can define project-specific target vocabularies.

## What Is Being Measured

The module does not estimate total language quantity. It asks whether and how often words from a target lexicon appear in a sample. Resources define base forms and accepted variants, so observed tokens can be mapped back to the intended target vocabulary.

Summary outputs include counts such as tokens and matched base forms, coverage values such as `lexicon_coverage`, and percentile fields when a resource declares usable norms.

## Built-In And Custom Resources

Built-in resources provide convenience for supported narratives. Custom resources let a project define its own targets, variants, and optional norm tables.

Structural similarity to CoreLex does not make every custom target vocabulary psychometrically validated. DIAAD validates resource format and consistency. Users remain responsible for whether a custom resource is appropriate for the task, population, and interpretation.

## Rates And Norms

Rates normalize count-like target-vocabulary fields by speaking time. Percentile fields depend on declared norm data and should be interpreted only when the source, columns, and comparison groups are appropriate.

## Draft Review Notes

Before publication, review built-in resource claims, CoreLex-related citations, custom-resource cautions, and norm/percentile wording.
