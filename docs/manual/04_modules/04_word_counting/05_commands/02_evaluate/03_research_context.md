# `words evaluate` Research Context

Word-count reliability evaluation asks whether two coding passes produced sufficiently similar utterance-level counts for the project's research purpose. DIAAD reports the comparison, but the project still has to decide how much disagreement is acceptable.

## What The Command Measures

The command compares paired primary and reliability counts by utterance. It reports the absolute difference, percent difference, percent similarity, a binary agreement flag, and an ICC summary across paired utterances.

The agreement flag is intentionally tolerant of small disagreements. A one-word difference can matter less for a long utterance than for a very short utterance, so DIAAD accepts agreement when the counts are within one word or when percent similarity is at least 85%.

## Interpretation Cautions

The one-word and 85-percent rules are operational thresholds, not universal methodological guarantees. A project with very short utterances, strict lexical inclusion rules, or high-stakes downstream analyses may need closer review of individual disagreements.

Percent similarity is easy to read, but it can behave sharply for short utterances. ICC is useful as a summary of paired numeric ratings, but it can be unstable or undefined when there are too few paired rows or little count variability.

Use the output as evidence for review and adjudication. It should be interpreted alongside coder training records, the project's word-counting rules, and any planned correction or adjudication process.

## Relation To Manual Coding

The evaluator cannot judge whether either coder applied the correct word-counting protocol. It can only compare the numbers entered in the workbooks. When reliability is low, inspect the utterance-level results and the original transcript context before deciding whether the problem is coder disagreement, ambiguous protocol language, utterance segmentation, or data-entry error.
