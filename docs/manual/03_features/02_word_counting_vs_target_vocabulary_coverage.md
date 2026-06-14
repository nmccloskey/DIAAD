# Word Counting Versus Target Vocabulary Coverage

DIAAD includes both manual word-count workflows and Target Vocabulary Coverage workflows. They both count language in some sense, but they answer different research questions and produce different kinds of measurements.

Word counting asks how many countable words a sample contains. Target Vocabulary Coverage asks how much of a configured target lexicon appears in a sample, often with base forms, acceptable variants, and coverage-style metrics.

This distinction matters early because the workflows can sit near each other in a monologic narrative analysis, but their outputs should not be interpreted as interchangeable "word count" measures.

## The Short Version

| Question | Word Counting | Target Vocabulary Coverage |
|---|---|---|
| Main construct | Total countable words. | Coverage of a predefined target vocabulary. |
| Typical unit | Utterance-level human-reviewed counts, summarized by sample. | Sample-level coverage and detail rows by target base form. |
| Human role | Human review is expected when coding rules require selective omissions. | Human design/review is required for custom resources; analysis is automated once resources and inputs are valid. |
| Lexicon | No fixed target lexicon. | Built-in or custom resource defines base forms and accepted variants. |
| Key outputs | `word_counting.xlsx`, reliability files, `word_counting_by_sample.xlsx`, rate outputs. | `target_vocab_data_YYMMDD_HHMM.xlsx` with `summary` and `details` sheets, plus optional rate outputs. |
| Rate meaning | Total words per minute when paired with speaking time. | Target-vocabulary count-like fields per minute when paired with speaking time. |

## Word Counting

The word-counting module supports manual or human-reviewed word counts. DIAAD can create a first-pass count from transcript-derived text, but that first pass is not a substitute for the project's coding rules.

This matters in discourse analysis because a project's word-count protocol may require selective omissions. For example, some research protocols exclude repetitions, part-word repetitions, nonword fillers, neologisms, prompt-only responses, or commentary outside the target sample. Those decisions depend on the coding standard and often require human judgment.

Operationally, `words files` creates coding and reliability workbooks. When Complete Utterance coding output is available, word-count file generation can use it so neutral or non-countable utterances are treated differently from ordinary countable utterances. Otherwise, transcript tables can serve as the fallback input.

After coding is reviewed, `words analyze` summarizes the completed word-count workbook. `words rates` can then combine sample-level word-count summaries with speaking-time values to calculate words per minute.

## Target Vocabulary Coverage

Target Vocabulary Coverage is lexicon driven. DIAAD loads built-in target vocabulary resources and can also load custom JSON resources from `advanced.target_vocabulary_resource_path`.

A resource defines:

- a stable resource identifier;
- display metadata such as language and task type;
- target base forms;
- accepted variants that map observed tokens back to base forms;
- optional norm-table specifications for percentile lookup.

During analysis, DIAAD reformats transcript-derived text, identifies target vocabulary matches, and writes a timestamped workbook under the `target_vocab/` output directory. The summary sheet includes fields such as `num_tokens`, `num_base_forms_produced`, `num_core_token_matches`, `lexicon_coverage`, `core_tokens_per_min`, and percentile fields when norms are available. The detail sheet records base-form-level counts and scores.

Built-in resources provide a CoreLex-style convenience layer for supported narrative stimuli. Custom resources generalize the same structure to project-specific target vocabularies, but DIAAD does not validate the psychometric quality of a custom lexicon. It checks resource structure and consistency; users remain responsible for whether the target vocabulary is theoretically and empirically appropriate.

## Why These Are Not Interchangeable

A sample with many words may still have low target-vocabulary coverage if it does not include the configured targets. A sample with strong target-vocabulary coverage may still be short. A rate column does not make the constructs identical; it only normalizes selected numerators by speaking time.

The safest interpretation is:

- word counts measure amount of countable language under a coding protocol;
- Target Vocabulary Coverage measures production of a predefined lexical target set;
- speaking-time rates normalize whichever count-like measure is being used;
- coverage proportions and percentile fields require their own interpretation.

## Read Next

- Functional overview: `docs/manual/01_overview/03_functional_overview.md`
- Configuration: `docs/manual/02_operation/04_configuration.md`
- Generated Example I/O: `docs/manual/03_features/04_generated_example_io.md`

Later Word Counting and Target Vocabulary Coverage module pages will provide command-specific inputs, outputs, reliability guidance, custom-resource schema details, and research-context discussion.

## Draft Review Notes

Before publication, review the methodological wording around manual word-count rules, CoreLex-style validity claims, and bespoke target-vocabulary resources. This page should remain clear that DIAAD validates resource format, not the psychometric adequacy of a custom lexicon.
