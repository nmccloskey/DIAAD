# Transcription Reliability Research Context

Transcription reliability asks whether two transcription passes produce sufficiently similar text for the project's purposes. DIAAD provides quantitative evidence and alignment artifacts, but the project still needs a transcription protocol and a review policy.

## Whole-Transcript Comparison

DIAAD evaluates transcription reliability at the level of processed transcript text. This is broader than checking only word totals or utterance segmentation. It can include words, nonwords, fillers, and repeated material after configured transcript processing.

This approach is useful because downstream discourse measures often depend on small transcript details. A missing repetition, different nonword spelling, or excluded speaker tier can affect later coding or counts.

## Character-Level Metrics

Levenshtein edit distance measures the number of single-character edits needed to transform one transcript into another. DIAAD reports a length-normalized Levenshtein similarity score:

```text
S_L = 1 - D_L / C_max
```

where `D_L` is the Levenshtein distance and `C_max` is the length of the longer processed transcript in characters.

Needleman-Wunsch global alignment provides another view of sequence similarity and produces alignment files that users can inspect manually.

## Why Character-Level Alignment

Character-level alignment is useful because it is algorithmically tractable and can be more tolerant of small spelling differences than word-level exact matching. For example, a transcript with a few typos may still score highly when most characters align.

The same feature also creates an interpretation boundary. A high character-level score does not prove that every linguistically important detail is correct, and a low score needs inspection before being treated as a final judgment.

## Practical Bands

DIAAD's report groups Levenshtein similarity values into practical bands. The working interpretation used in the research notes treats values near 0.7 as minimal, 0.8 as sufficient, and 0.9 as excellent.

These bands should be reported cautiously. Levenshtein similarity is not the same as ICC or Cronbach's alpha. It is a standardized text-similarity metric that can serve a similar auditing purpose, but more systematic work is needed to establish universal thresholds.

## Interpreting Low Scores

For low or surprising scores, inspect the alignment file. Common causes include:

- missing transcript sections;
- differences in repetitions;
- nonword spelling differences;
- speaker tiers that should or should not have been excluded;
- CLAN correction or markup handling;
- metadata matching errors that paired the wrong files.

## Draft Review Notes

Before publication, review the practical Levenshtein similarity bands, citation language, and reporting recommendations against the final references and intended methodological claims.

## Read Next

- `transcripts evaluate` research context: `docs/manual/04_modules/01_transcripts/05_commands/04_evaluate/03_research_context.md`
- Transcript text normalization research context: `docs/manual/05_functionalities/07_transcript_text_normalization_speaker_exclusion/03_research_context.md`
- Reliability functionality research context: `docs/manual/05_functionalities/12_reliability_selection_evaluation_reselection/03_research_context.md`
