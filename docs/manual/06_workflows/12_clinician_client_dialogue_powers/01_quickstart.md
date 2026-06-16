# Clinician-Client Dialogue POWERS Quickstart

Use this workflow when clinician-client or other dialogic transcript samples will be coded with POWERS. DIAAD supports file generation, first-pass automation for selected fields, human review, reliability evaluation, optional reselection, analysis, and rates.

## Starting Point

Begin from the shared transcription baseline:

```bash
diaad transcripts tabularize
```

Before generating POWERS files, inspect the transcript table for stable sample identifiers, utterance identifiers, speaker labels, row order, and any content that affects privacy or de-identification decisions.

## Core Sequence

Generate POWERS coding and reliability workbooks:

```bash
diaad powers files
```

Complete human review and manual coding. Automation is only first-pass support for selected fields.

After primary and reliability workbooks are complete, evaluate reliability:

```bash
diaad powers evaluate
```

If another reliability subset is needed, run:

```bash
diaad powers reselect
```

Analyze completed coding:

```bash
diaad powers analyze
```

Create a speaking-time template, enter speaking time in seconds, and calculate rates:

```bash
diaad templates times
diaad powers rates
```

## Privacy Checkpoint

Clinician-client dialogs may contain explicit or implicit personal information. When de-identification is impractical, a local CLI workflow may be preferable to hosted web use. Local Streamlit can also be useful when users want an interface while keeping files on their own machine.

## Read Next

- Shared transcription baseline: `docs/manual/06_workflows/04_transcription_based_workflow_baseline/02_usage_guide.md`
- POWERS module: `docs/manual/04_modules/05_powers/01_quickstart.md`
- POWERS automation support: `docs/manual/05_functionalities/17_powers_automation_support/02_usage_guide.md`
- Blinding functionality: `docs/manual/05_functionalities/14_blinding_unblinding_auto_blind/02_usage_guide.md`
