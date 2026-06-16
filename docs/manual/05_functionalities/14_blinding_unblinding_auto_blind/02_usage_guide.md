# Blinding, Unblinding, and Auto-Blind Usage Guide

Blinding is useful when a project wants coders or analysts to work with masked identifiers. It is also easy to misuse, so plan where encoded and decoded files belong in the workflow before creating them.

## Encode Before Manual Coding

For manual coding, prepare the coding workbook first, then encode identifiers before distributing coder-facing files.

You can do this with the standalone command:

```bash
diaad blinding encode
```

or with supported coding workflows that apply `auto_blind` during file generation.

When blinding is active for coder-facing files, DIAAD may replace configured identifier columns in place with blind codes. This keeps the workbook usable for coders while hiding the raw values.

## Decode Before DIAAD Analysis

After manual coding is complete, decode back to canonical identifiers before DIAAD analysis when possible:

```bash
diaad blinding decode
```

This is usually safer because many analysis workflows expect sample identifiers to match transcript tables, metadata, codebooks, or other project files. Decoding also makes audit review easier: the analysis output can be traced back to the canonical sample frame.

## Encode Again After Analysis If Needed

After analysis, a project may encode selected exports again for blinded statistical workflows or external collaboration. This is a separate sharing decision. Preserve the decoded canonical analysis files and the codebook in controlled storage.

If blinded values need to stay comparable across multiple files, reuse the same codebook.

## Standalone Encode And Decode

Standalone `blinding encode` expects an input directory with one target `.xlsx` workbook. If a blind codebook is present, DIAAD reuses it. If no codebook is found, DIAAD generates one from configured `blind_columns`.

Standalone `blinding decode` expects one target `.xlsx` workbook plus one codebook. It can decode suffixed analysis-style columns such as `sample_id_blinded` and in-place blinded columns such as a `sample_id` column containing blind codes.

Use a dedicated input directory for each encode/decode step so file discovery is unambiguous.

## What Blinding Does Not Do

Blinding only masks configured columns. It does not:

- remove identifying transcript content;
- remove identifying free-text comments;
- inspect filenames embedded inside workbook cells;
- prevent coders from recognizing samples from memory or context;
- replace project-level de-identification, access control, or data-governance review.

For highly sensitive or identifying content, local CLI workflows may be preferable to hosted web workflows, even when the hosted workflow uses temporary processing and returns outputs as a ZIP.

## Read Next

- Blinding research context: `docs/manual/05_functionalities/14_blinding_unblinding_auto_blind/03_research_context.md`
- Exact file name matching: `docs/manual/03_features/03_exact_file_name_matching.md`
- Configurable identifiers: `docs/manual/05_functionalities/10_configurable_sample_utterance_identifiers/02_usage_guide.md`
- Web app operation: `docs/manual/02_operation/03_webapp.md`
