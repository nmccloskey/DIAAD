# Target Vocabulary Resource Management Research Context

Target Vocabulary Coverage is a resource-driven method. The resource is not just a technical input file; it encodes the target lexicon, acceptable variants, and any norm reference that shapes how coverage is interpreted.

## CoreLex-Like But General

DIAAD includes CoreLex-style narrative resources as a convenience layer for established narrative tasks. The broader functionality is more general: a project can define any task-specific target vocabulary and run the same coverage machinery over transcript-derived utterances.

This makes the module useful beyond canonical CoreLex-style stimuli, but it also shifts responsibility to the project. A custom resource may be structurally valid while still needing independent evidence that the target words, accepted variants, and resulting metrics are meaningful for the population and task.

## Resource Design Is An Analytic Decision

Choices in `base_forms` and `variant_map` define what can count as lexical success. Those choices affect:

- how strict the measure is;
- whether inflected or colloquial forms are accepted;
- whether repeated target words contribute to token-match counts;
- whether coverage reflects a narrow stimulus protocol or a broader semantic target set.

For transparent reporting, describe how the target vocabulary was selected, how variants were chosen, and whether the resource was adapted from an established protocol.

## Norms And Percentiles

Resources can declare norm tables that DIAAD loads during analysis. Built-in CoreLex-style resources declare online CSV norm sources. DIAAD retrieves those tables and computes percentile fields locally; it does not upload transcript data to the norm source.

Norm availability is not the same as norm applicability. Percentiles should be reported only when the norm source, participant group, task, and metric definition match the project's intended comparison.

## Validation Boundary

The module-level Target Vocabulary Coverage research context describes the built-in CoreLex-style validation notes and related acknowledgments. Those implementation-level validation checks should not be generalized automatically to every custom target-vocabulary resource.

## Draft Review Notes

Before publication, review the CoreLex, Dalton, Pritchard, and Cavanaugh citation framing against the final references section. Also review any validation figures retained in the module-level Target Vocabulary Coverage page.

## Read Next

- Target Vocabulary Coverage research context: `docs/manual/04_modules/06_target_vocabulary_coverage/03_research_context.md`
- Introduction and acknowledgments: `docs/manual/01_overview/01_introduction.md`
- Generated Example I/O: `docs/manual/03_features/04_generated_example_io.md`
