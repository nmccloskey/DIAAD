# Monologic Narrative Complete Utterances Usage Guide

Complete Utterance coding is a transcript-table workflow. DIAAD creates the workbook structure and downstream summaries, but the coding judgments come from the project's CU protocol.

## Prepare The Transcript Table

Start from the shared transcription baseline:

```bash
diaad transcripts tabularize
```

Before generating CU files, inspect:

- utterance text;
- speaker labels;
- metadata and stimulus fields;
- `position` and `position_sub`;
- any transcript-table revisions that may affect utterance identity.

Configure `project.exclude_speakers` before file generation if investigator, clinician, or other non-target speaker rows should be marked not applicable for CU coding.

## Configure Coding Structure

Important settings include:

| Setting | Use |
|---|---|
| `project.reliability_fraction` | Sample fraction for reliability workbook. |
| `project.num_coders` | Coder assignment and primary/reliability coding layout. |
| `advanced.cu_paradigms` | Optional paradigm-specific SV/REL columns. |
| `advanced.auto_blind` | Optional blinding for coder-facing files. |

`advanced.cu_paradigms` can support project-specific variants of CU judgment. For example, a project may use separate columns for dialect-sensitive coding rules. This is a configurable example, not a universal requirement.

## Generate Coding Workbooks

Run:

```bash
diaad cus files
```

The primary workbook contains utterance-level rows and CU coding columns. The reliability workbook contains selected reliability material based on `project.reliability_fraction`.

If the project uses blinding, encode identifiers before distributing coder-facing workbooks. Decode back to original sample identifiers before DIAAD analysis when downstream joins require them.

## Complete Coding And Reliability

Coders complete the CU fields according to the project protocol. In the basic schema, DIAAD later derives CU from:

```text
sv
rel
```

CU is positive only when both grammaticality or structure and relevance criteria are coded positive in the detected SV/REL pair.

Run reliability evaluation after both primary and reliability coding are complete:

```bash
diaad cus evaluate
```

If the reliability results indicate that another round is needed, use reselection as a documented fallback:

```bash
diaad cus reselect
```

## Analyze Completed Coding

After coding has been reviewed, run:

```bash
diaad cus analyze
```

The analysis adds derived CU columns and writes utterance-level, long sample-level, and wide sample-level summaries.

Inspect rows with missing or inconsistent SV/REL values before treating the summary as final.

## Add Rates

If speaking-time-normalized CU measures are needed, create the speaking-time template:

```bash
diaad templates times
```

Enter speaking time in seconds, then run:

```bash
diaad cus rates
```

Rates are per minute. They should be interpreted alongside raw CU counts and the project's speaking-time measurement procedure.

## Read Next

- Shared transcription baseline: `docs/manual/06_workflows/04_transcription_based_workflow_baseline/02_usage_guide.md`
- `cus evaluate`: `docs/manual/04_modules/03_complete_utterances/05_commands/02_evaluate/02_usage_guide.md`
- `cus analyze`: `docs/manual/04_modules/03_complete_utterances/05_commands/04_analyze/02_usage_guide.md`
- Revision handling: `docs/manual/05_functionalities/11_revision_handling/02_usage_guide.md`
