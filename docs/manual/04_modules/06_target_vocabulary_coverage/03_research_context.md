# Target Vocabulary Coverage Research Context

Target Vocabulary Coverage, or TVC, measures how much of a predefined lexical target set appears in a language sample. The module is CoreLex-like in structure, but its main design is more general: a resource can define any target vocabulary, accepted variants for each target, and optional norm tables.

## What Is Being Measured

TVC does not estimate total language quantity. It asks whether and how often words from a target lexicon appear in a sample.

Each resource defines base forms and accepted variants. During analysis, observed tokens are normalized and mapped back to base forms. The main summary fields include:

```text
num_tokens
num_base_forms_produced
num_core_token_matches
lexicon_coverage
core_tokens_per_min
```

`num_base_forms_produced` counts how many distinct target base forms were produced. `num_core_token_matches` counts how many observed tokens matched the target vocabulary, including repeated matches. `lexicon_coverage` is the proportion of the resource's base forms that appeared at least once.

This distinction matters because two samples can contain the same number of matched target tokens while covering different proportions of a lexicon, and different stimuli can have target lexicons of different sizes.

## CoreLex As A Built-In Layer

Core Lexicon analysis has demonstrated concurrent validity and reliability and has proven amenable to automation (Kim & Wright, 2020; Dalton et al., 2022). DIAAD's TVC module includes five built-in CoreLex-style narrative resources:

```text
BrokenWindow
CatRescue
Cinderella
RefusedUmbrella
Sandwich
```

These built-ins provide a convenience layer for canonical narrative tasks. They are not the only intended use of the module. DIAAD extends the same resource structure to user-defined target vocabularies, so projects can analyze task-specific lexicons for prompts, procedures, intervention targets, or other elicitation contexts.

DIAAD is not positioned as a replacement for established standalone CoreLex tools such as Cavanaugh et al.'s web application; its contribution is integration: batch processing, database-friendly tables, CLI and web app access, custom resources, aggregate coverage metrics, and rate-ready outputs for downstream analysis. The Introduction includes acknowledgments for the coreLexicon project and related open-source resources.

## Validation And Scope

In validation work comparing a functionally equivalent DIAAD implementation with Cavanaugh et al.'s CoreLex web application, 402 samples comprising Broken Window, Cat Rescue, and Refused Umbrella stimuli were analyzed with both systems. ICC(2) values for primary metrics were high: 0.963 for number of core words and 0.969 for core words per minute.

Those figures support DIAAD's implementation for the validated CoreLex-style use case described in the notes. They should not be generalized automatically to every custom TVC resource. A bespoke target vocabulary may be structurally similar to CoreLex while still requiring independent psychometric evaluation for a particular task, population, and interpretation (Pritchard et al., 2018).

## Built-In And Custom Resources

Built-in resources require no user JSON. Custom resources let a project define:

```text
resource_id
display_name
language
task_type
base_forms
variant_map
norms
```

DIAAD validates resource structure and consistency. It checks that required fields are present, base forms are unique, variants map cleanly to base forms, and declared norm specifications have the expected shape. This is format validation, not evidence that the resource is clinically or psychometrically valid.

When a custom resource path is configured, built-in resources remain available. If a custom resource uses the same resource ID as a built-in resource, the custom resource overrides that built-in definition for the run.

## Rates And Norms

Rates normalize count-like TVC fields by speaking time. `vocab analyze` also computes `core_tokens_per_min` directly from the analysis input when speaking time is available, and `vocab rates` can add additional per-minute columns from the analysis summary.

Percentile fields depend on norm tables declared by the active resource. Built-in CoreLex-style resources declare online CSV norm sources. Loading those norms requires network access, but the analysis does not upload transcript data; it retrieves resource-declared norm tables and computes percentiles locally.

Percentiles should be interpreted only when the norm source, comparison groups, metric definitions, and sample task match the project's research question.

## Read Next

- Introduction and acknowledgments: `docs/manual/01_overview/01_introduction.md`
- Word Counting Versus Target Vocabulary Coverage: `docs/manual/03_features/02_word_counting_vs_target_vocabulary_coverage.md`
- Target Vocabulary Coverage quickstart: `docs/manual/04_modules/06_target_vocabulary_coverage/01_quickstart.md`
- Generated Example I/O: `docs/manual/03_features/04_generated_example_io.md`
