# POWERS Automation Support Research Context

POWERS automation should be understood as a human-in-the-loop support layer. It can make coding more efficient, especially for text-derived counts, but it does not resolve interpretive POWERS coding decisions.

## Why Automation Is Limited

Some POWERS fields have a relatively direct relationship to transcript text. For example, filled pauses and selected token-count measures can be estimated from utterance strings with rule-based and NLP-assisted procedures.

Other fields depend more heavily on human interpretation, conversational context, and the POWERS coding manual. DIAAD therefore leaves fields such as `turn_type`, paraphasia-related fields, circumlocutions, comments, large pauses, and collaborative repair for human coding.

## Human Review

Automated first-pass values can still be wrong. Errors may come from transcript conventions, CHAT markers, disfluencies, spelling, utterance segmentation, speaker labels, spaCy model behavior, or cases where the POWERS protocol treats a form differently than a general NLP model would.

For that reason, automated values should be reviewed before:

- coder-facing workbooks are treated as complete;
- reliability is evaluated;
- analysis summaries are generated;
- rates are interpreted.

## Reliability Interpretation

Reliability statistics can be misleading when both coders inherit identical automated fields and do not review them independently. High agreement for automated fields may reflect a shared prefill rather than coder consistency.

Projects should decide whether coders review automated values independently, whether corrections are tracked, and how automated fields are described in methods reporting.

## Section E Boundary

Section E fields are currently best understood as sample-level note or descriptor fields. They are generated with the POWERS coding workbook, but DIAAD does not currently summarize, evaluate, or rate them like the main utterance-level POWERS variables.

## Draft Review Notes

Before publication, review this page against the POWERS manual and any final methods language. The automation description should stay balanced: useful as first-pass support, but not evidence of validated automated POWERS coding.

## Read Next

- POWERS research context: `docs/manual/04_modules/05_powers/03_research_context.md`
- `powers evaluate` research context: `docs/manual/04_modules/05_powers/05_commands/04_evaluate/03_research_context.md`
- POWERS automation implementation notes: `docs/manual/05_functionalities/17_powers_automation_support/04_implementation_notes.md`
