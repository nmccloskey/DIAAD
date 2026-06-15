# `vocab analyze` Research Context

`vocab analyze` produces evidence about coverage of a defined target lexicon. Its outputs are meaningful only in relation to the resource being used.

## CoreLex-Style Resources

For bundled CoreLex-style resources, the target vocabulary corresponds to a canonical narrative resource. Metrics such as `num_base_forms_produced`, `num_core_token_matches`, `core_tokens_per_min`, and percentile fields should be interpreted in relation to the corresponding CoreLex literature and norm source.

The module-level research context summarizes DIAAD's broader CoreLex/TVC framing and validation figures.

## Custom Resources

For custom resources, DIAAD can compute the same fields, but the interpretation changes. A custom resource might be useful for a treatment target list, procedural description, discussion prompt, or project-specific elicitation task. That does not mean the resulting metric has the same validation status as CoreLex.

The resource check confirms that a custom JSON file is structurally valid. It does not establish that the lexicon is complete, clinically meaningful, developmentally appropriate, or comparable across samples.

## Coverage Versus Quantity

`lexicon_coverage` is a normalized coverage measure: distinct target base forms produced divided by the total number of base forms in the resource. It is not the same as total word count or language quantity.

`num_core_token_matches` counts target-token matches and can increase when a speaker repeats the same target item. `num_base_forms_produced` and `lexicon_coverage` instead emphasize breadth of target vocabulary coverage.

## Percentiles And Norms

Percentile fields are computed only when a resource declares norm data and those data can be loaded. Built-in resources may declare online norm sources, which means percentile computation can depend on network access. Transcript data are not uploaded for this step.

Blank percentile fields do not necessarily mean coverage failed. They may mean that no applicable norm table was declared or available.

## Read Next

- Target Vocabulary Coverage research context: `docs/manual/04_modules/06_target_vocabulary_coverage/03_research_context.md`
- Introduction and acknowledgments: `docs/manual/01_overview/01_introduction.md`
- Word Counting Versus Target Vocabulary Coverage: `docs/manual/03_features/02_word_counting_vs_target_vocabulary_coverage.md`
