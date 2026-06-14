# POWERS Implementation Notes

The POWERS module is implemented under `src/diaad/coding/powers/`.

## File Generation

`powers files` reads transcript tables, validates configured identifiers, prepares utterance-level coding rows, assigns primary and reliability coding material, and writes outputs under `powers_coding/`.

The primary workbook contains an `utterance_coding` sheet and a `section_e` sheet. The command can also write a reliability workbook and a blind codebook when blinding is configured.

## Automation

POWERS automation is controlled by `project.automate_powers` and the configured `advanced.spacy_model_name`. The default model name is `en_core_web_sm`.

Automation loads NLP support, computes selected text-derived fields, and falls back to manual completion when automation is unavailable or disabled. Users should still review generated values.

## Analysis, Reliability, And Rates

`powers analyze` reads completed POWERS coding and writes `powers_analysis.xlsx` under `powers_coding_analysis/`.

`powers evaluate` compares primary and reliability workbooks, producing merged data, continuous summaries, categorical summaries, and a report.

`powers rates` reads POWERS analysis summaries and speaking-time values, then writes `powers_coding_rates.xlsx`.

## Boundaries

The implementation is workbook-centered and identifier-driven. Command pages should document exact sheets and required columns after source-level review.
