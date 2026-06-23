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

## Team-Specific Coding Rules

For word counts and selected POWERS metrics, DIAAD's automation is best understood as first-pass coding for human review. It does not currently automate the complete utterance coding paradigm. Complete utterance coding requires context-dependent judgments about grammaticality and relevance that would need a specialized model and a carefully validated protocol.

Even count-like fields may require project-specific rules. For example, a team may need to decide whether to omit word and part-word repetitions, neologisms, nonword fillers, opening or closing commentary, or direct yes/no responses that add no content beyond a clinician prompt. Published protocols may provide the starting point, but they may not resolve every case in a dataset.

Projects using POWERS automation should therefore document the rule set used for manual review, coder training, adjudication, and methods reporting. This is especially important when adapting POWERS to local conventions or bespoke extensions of the coding paradigm.

## Reliability Interpretation

Reliability statistics can be misleading when both coders inherit identical automated fields and do not review them independently. High agreement for automated fields may reflect a shared prefill rather than coder consistency.

Projects should decide whether coders review automated values independently, whether corrections are tracked, and how automated fields are described in methods reporting.

## Validation Example From The RASCAL Archive

RASCAL archived workflow no. 2 documents a lab-specific validation of POWERS automation from a deprecated DIAAD precursor system. The workflow compared automated and manual coding for 36 of 181 clinician-client dialog samples, giving 19.9% coverage. Sampling was stratified by cycle, site, and test; in that dataset, cycle also separated groups with more severe versus milder aphasia profiles.

The archived workflow used regex and spaCy `en_core_web_trf`. DIAAD's current default is the lighter `en_core_web_sm`, so the archived figures should not be treated as expected performance for current DIAAD runs.

| Automated field | ICC2, severe profiles | ICC2, mild profiles |
|---|---:|---:|
| `speech_units` | 0.9981 | 0.9998 |
| `content_words` | 0.7645 | 0.9078 |
| `num_nouns` | 0.5087 | 0.8018 |
| `filled_pauses` | 0.9786 | 0.9873 |

These results suggest that automated first-pass values can be helpful in manual coding workflows, especially for speech-unit and filled-pause counts. They also show why human review remains necessary: noun-count agreement was much lower for the more severe cohort.

The figures are not a measure of spaCy model accuracy. They reflect alignment between one lab's manual coding rules, one dataset, one spaCy model choice, and the particular way spaCy is used in `src/diaad/coding/powers/automation.py`. Other datasets, transcript conventions, model choices, or team-specific coding rules may produce different results.

For full procedural details, see the RASCAL archive: [POWERS Automation Validation 2025](https://github.com/nmccloskey/RASCAL/tree/main/archived_workflows/02_powers_automation_validation_2025).

## Current DIAAD Validation Pattern

The deprecated precursor supported automation validation more directly than current DIAAD. In current DIAAD, a comparable validation can be assembled with the standard subset, file-generation, and evaluation commands:

1. Use `diaad templates subset` to randomly select validation samples.
2. Prepare one POWERS coding file set with `diaad powers files` and `automate_powers: true`.
3. Prepare a second POWERS coding file set from the same samples with `automate_powers: false`.
4. Manually complete the blank file set.
5. Treat one file set as the reliability set and run `diaad powers evaluate` to calculate agreement statistics such as ICC(2,1).

When reporting this kind of comparison, describe it as agreement between an automated first pass and a human-reviewed or manually coded reference, not as evidence that the automated fields are final analysis-ready values.

## Section E Boundary

Section E fields are currently best understood as sample-level note or descriptor fields. They are generated with the POWERS coding workbook, but DIAAD does not currently summarize, evaluate, or rate them like the main utterance-level POWERS variables.

## Research Reporting Notes

Automation descriptions should stay balanced: useful as first-pass support, but not evidence of fully validated automated POWERS coding for every dataset. Methods text should identify which fields were automated, which fields were manually coded, what review protocol was used, which spaCy model was configured, and how automation-informed fields entered reliability evaluation.

## Read Next

- POWERS research context: `docs/manual/04_modules/05_powers/03_research_context.md`
- `powers evaluate` research context: `docs/manual/04_modules/05_powers/05_commands/04_evaluate/03_research_context.md`
- POWERS automation implementation notes: `docs/manual/05_functionalities/17_powers_automation_support/04_implementation_notes.md`
