# Reliability Selection, Evaluation, and Reselection Research Context

Reliability workflows help a project evaluate whether transcript or coding procedures can be applied consistently. DIAAD reports structured comparisons, but the project still needs a protocol, coder training, review procedures, and interpretation standards.

## Reliability Is Protocol-Specific

Different DIAAD modules compare different evidence. A transcription reliability score summarizes similarity between two transcript texts. A word-count reliability report compares numeric counts for paired utterances. A POWERS report compares a mix of continuous and categorical variables. A Digital Conversational Turns report compares counts and turn sequences.

Because the constructs differ, reliability thresholds should not be treated as universal across modules. DIAAD can calculate metrics and surface disagreements; it cannot decide that a threshold is appropriate for every research question.

## Transcription Reliability

DIAAD's transcription reliability evaluation compares whole transcript text after configured normalization. It reports token and character count differences, Levenshtein edit-distance metrics, Needleman-Wunsch global alignment metrics, and alignment files for manual inspection.

The report groups Levenshtein similarity into practical bands. Those bands are aids for review, not a substitute for project-specific judgment about what transcript differences matter.

## Manual Coding Reliability

Manual coding reliability is tightly linked to the coding manual. If coders disagree, the cause may be coder error, an ambiguous coding rule, a transcript problem, or an inadequately operationalized construct. The detailed output should therefore be used diagnostically, not only as a pass/fail summary.

For semi-automated workflows, automated first passes should still be reviewed by humans before reliability evaluation. Reliability of unrevised first-pass data may describe the automation and data-entry state rather than the intended coding protocol.

## Reselection As A Fallback

Reselection supports another reliability round after the first selected subset is insufficient. It should be documented as part of the project's reliability history, especially if the final reported reliability depends on replacement samples.

## Review Note

TODO: Before publication, review whether the transcription reliability similarity bands should be presented with citation-backed thresholds, implementation-only wording, or project-specific cautionary language.

## Read Next

- Methodological overview: `docs/manual/01_overview/02_methodolgical_overview.md`
- Transcript text normalization: `docs/manual/05_functionalities/07_transcript_text_normalization_speaker_exclusion/03_research_context.md`
- Revision handling: `docs/manual/05_functionalities/11_revision_handling/03_research_context.md`
