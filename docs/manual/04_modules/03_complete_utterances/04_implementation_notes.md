# Complete Utterances Implementation Notes

The Complete Utterances module is implemented under `src/diaad/coding/compl_utts/`.

## File Generation

`cus files` reads transcript tables and prepares CU coding exports. It supports configurable sample and utterance identifier columns, reliability fraction, random seed, coder assignment, and optional blinding.

The primary workbook is written as `cu_coding.xlsx`. A reliability workbook is written when the configured reliability fraction selects eligible rows. If blinding is active, the workflow can also write a blind codebook.

## Analysis

`cus analyze` reads completed CU coding workbooks, detects coder/paradigm column pairs, computes utterance-level and sample-level summaries, and writes long and wide sample summaries under `cu_coding_analysis/`.

When auto-blinding resources are available, analysis can reconnect blinded identifiers before writing outputs.

## Reliability And Reselection

`cus evaluate` merges primary and reliability files on configured identifiers and summarizes agreement. `cus reselect` uses shared reselection utilities to build a replacement reliability workbook while avoiding previously used material where possible.

## Rates

`cus rates` reads the configured CU sample summary file and the configured speaking-time file, converts speaking time to minutes, and writes `cu_coding_rates.xlsx`.

## Boundaries

The implementation expects completed coding columns to follow DIAAD's generated workbook structure. Command pages should document exact column expectations after source-level review.
