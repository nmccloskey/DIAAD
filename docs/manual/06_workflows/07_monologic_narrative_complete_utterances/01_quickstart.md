# Monologic Narrative Complete Utterances Quickstart

Use this workflow when monologic narrative transcripts are ready for Complete Utterance coding. DIAAD organizes transcript-derived utterances into coding workbooks, supports reliability evaluation, summarizes completed coding, and calculates rates when speaking time is available.

## Starting Point

Begin from a reviewed transcript table:

```bash
diaad transcripts tabularize
```

Inspect `transcript_tables/transcript_tables.xlsx` before generating coding files.

## Core Sequence

Create coding and reliability workbooks:

```bash
diaad cus files
```

Complete manual CU coding outside DIAAD, then evaluate reliability:

```bash
diaad cus evaluate
```

Use reselection only if another reliability round is needed:

```bash
diaad cus reselect
```

After coding is complete and checked, analyze:

```bash
diaad cus analyze
```

If rates are needed, create or complete a speaking-time workbook and run:

```bash
diaad templates times
diaad cus rates
```

## Key Outputs

```text
cu_coding/cu_coding.xlsx
cu_coding/cu_reliability_coding.xlsx
cu_reliability/
cu_coding_analysis/
```

## Review Checkpoint

Complete Utterance coding is a manual protocol. Reliability is strongly recommended for research workflows, but DIAAD does not decide which threshold is sufficient for every project.

## Read Next

- Complete Utterances module: `docs/manual/04_modules/03_complete_utterances/01_quickstart.md`
- `cus files`: `docs/manual/04_modules/03_complete_utterances/05_commands/01_files/01_quickstart.md`
- Reliability functionality: `docs/manual/05_functionalities/12_reliability_selection_evaluation_reselection/02_usage_guide.md`
- Speaking-time rates: `docs/manual/05_functionalities/15_speaking_time_rate_calculation/02_usage_guide.md`
