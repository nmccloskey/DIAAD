# Monologic Narrative Integrated Workflow Usage Guide

The integrated monologic workflow is a project-level path through several DIAAD modules. It is useful when a research group wants narrative samples to support multiple related measures, such as complete utterances, reviewed word counts, target-vocabulary coverage, and per-minute rates.

This page assumes the shared transcription baseline has already produced a reviewed transcript table.

## Stage 1: Stabilize The Transcript Table

Before generating any manual coding files, check:

- sample identifiers;
- stimulus or narrative labels;
- participant and examiner speaker labels;
- utterance text;
- `position` and `position_sub`;
- metadata parsed from filenames or transcript headers;
- any transcript revisions that may affect utterance identity or row order.

If transcripts are still being revised, delay coding-file generation until the project can identify the transcript table version that downstream coding will use. If a transcript table changes after coding has begun, decide whether affected coding files need to be regenerated, recoded, or reanalyzed.

## Stage 2: Plan Blinding

Manual coding workflows often benefit from encoded identifiers. A common pattern is:

1. generate the coder-facing workbook;
2. encode configured identifiers before distribution;
3. complete coding;
4. decode back to canonical identifiers before DIAAD analysis;
5. encode selected exports again if blinded statistical workflows require them.

This is an ideal pattern, not a requirement. Blinding does not remove identifying transcript content, and coders may still recognize a sample from its details.

## Stage 3: Complete Utterance Coding

Complete Utterance coding usually comes first because its analysis can provide utterance-level inclusion context for word-counting files.

Run:

```bash
diaad cus files
```

After primary and reliability coding are complete, run:

```bash
diaad cus evaluate
diaad cus analyze
```

Use `diaad cus reselect` only if the project needs another reliability round. Treat reliability as a review checkpoint tied to the project protocol, not as a universal pass/fail value supplied by DIAAD.

## Stage 4: Human-Reviewed Word Counting

After CU analysis, run:

```bash
diaad words files
```

When CU-derived utterance-level output is available, word-counting file generation can use it as a preferred input. This helps align word counting with prior inclusion decisions, such as excluded speakers or CU-neutral rows.

Review the first-pass word counts manually, complete the reliability workbook, then run:

```bash
diaad words evaluate
diaad words analyze
```

Use `diaad words reselect` only when another reliability subset is needed.

## Stage 5: Target Vocabulary Coverage

Run Target Vocabulary Coverage when the narrative task has an appropriate active resource. For bundled CoreLex-style tasks, check that stimulus values match active resource IDs. For project-specific tasks, create or select a custom resource and inspect it before analysis.

```bash
diaad vocab check
diaad vocab analyze
```

TVC is not a substitute for total word count. It asks whether a sample covers configured target vocabulary. Word Counting asks how much countable language the sample contains under a project protocol.

## Stage 6: Speaking Time And Rates

Create a speaking-time workbook:

```bash
diaad templates times
```

Enter speaking time in seconds. Then run the rate commands that match the completed analyses:

```bash
diaad cus rates
diaad words rates
diaad vocab rates
```

Rates are per minute, but each module uses a different numerator. Do not compare CU, word-count, and TVC rates as if they measured the same construct.

## Stage 7: Integration And Export

The integrated workflow is not a single merge command. Integration depends on stable sample identifiers, metadata fields, and a documented analysis plan. Preserve the intermediate outputs because they explain how final analysis tables were produced.

Common final artifacts include:

```text
cu_coding_analysis/
word_count_analysis/
target_vocab_analysis/
coding_templates/speaking_times.xlsx
```

If final exports should be blinded, encode those exports after analysis while preserving decoded canonical outputs and codebooks in controlled storage.

## Common Problems

If word-counting files do not reflect CU inclusion decisions, check whether `cu_coding_by_utterance.xlsx` is available in the active input or output tree.

If TVC coverage is unexpectedly low, check stimulus labels against active resource IDs and inspect the resource `variant_map`.

If rates look too small or too large, confirm that speaking-time values are in seconds, not minutes.

## Read Next

- Revision handling: `docs/manual/05_functionalities/11_revision_handling/02_usage_guide.md`
- Speaking-time rates: `docs/manual/05_functionalities/15_speaking_time_rate_calculation/02_usage_guide.md`
- Word Counting versus TVC: `docs/manual/03_features/02_word_counting_vs_target_vocabulary_coverage.md`
- Run provenance and audit artifacts: `docs/manual/05_functionalities/03_run_provenance_audit_artifacts/02_usage_guide.md`
