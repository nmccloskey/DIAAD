# Monologic Narrative Integrated Workflow Quickstart

Use this workflow when a project wants a coordinated monologic narrative pipeline rather than a single isolated measure. The usual sequence combines transcript tabularization, Complete Utterances, human-reviewed word counts, Target Vocabulary Coverage where the task supports it, speaking-time rates, and optional blinding for coding or export.

## Starting Point

Complete the shared transcription baseline first:

```bash
diaad transcripts tabularize
```

Inspect the transcript table before branching. The sample identifier, stimulus or narrative field, utterance text, speaker labels, and row order become shared context for later outputs.

## Core Sequence

Generate and complete Complete Utterance coding:

```bash
diaad cus files
diaad cus evaluate
diaad cus analyze
```

Use reselection only when another CU reliability round is needed:

```bash
diaad cus reselect
```

Generate and complete word-counting files, preferably after CU analysis so DIAAD can use CU-derived utterance inclusion information:

```bash
diaad words files
diaad words evaluate
diaad words analyze
```

Use reselection only when another word-count reliability round is needed:

```bash
diaad words reselect
```

Run Target Vocabulary Coverage when the narrative stimulus has an appropriate built-in or custom target-vocabulary resource:

```bash
diaad vocab check
diaad vocab analyze
```

Add speaking-time-normalized rates where appropriate:

```bash
diaad templates times
diaad cus rates
diaad words rates
diaad vocab rates
```

## Blinding Checkpoint

For manual coding, encoding identifiers before coder-facing workbooks are distributed is often ideal. Decode back to canonical sample identifiers before DIAAD analysis when downstream joins require original IDs. After analysis, a project may encode selected exports again for blinded statistical workflows.

## Read Next

- Shared transcription baseline: `docs/manual/06_workflows/04_transcription_based_workflow_baseline/02_usage_guide.md`
- Complete Utterances workflow: `docs/manual/06_workflows/07_monologic_narrative_complete_utterances/02_usage_guide.md`
- Word Counting workflow: `docs/manual/06_workflows/08_monologic_narrative_word_counting/02_usage_guide.md`
- Target Vocabulary Coverage workflow: `docs/manual/06_workflows/09_monologic_narrative_target_vocabulary_coverage/02_usage_guide.md`
- Blinding functionality: `docs/manual/05_functionalities/14_blinding_unblinding_auto_blind/02_usage_guide.md`
