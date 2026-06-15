# `turns evaluate` Research Context

DCT reliability has two related but distinct questions: whether coders assigned similar numbers of turns to each speaker, and whether they represented the sequence of turns similarly.

## Counts And Sequences

Count agreement is useful when the main question is speaker participation. Two coders may agree that participant `1` took five turns and participant `0` took six turns even if their turn strings place those turns in different positions.

Sequence agreement is useful when the order of turns matters. Levenshtein similarity is sensitive to inserted, deleted, or substituted characters in the turn string, so it can reveal disagreements that count summaries conceal.

## Interpreting Low Agreement

Low agreement may reflect several different coding issues:

- ambiguous speaker identity;
- inconsistent turn-boundary rules;
- inconsistent handling of overlap;
- invalid or project-inconsistent digit syntax;
- missing rows in one workbook.

These causes have different implications. For example, low count agreement may point to participation coding problems, while high count agreement with low sequence similarity may point to disagreement about turn order or segmentation.

## Thresholds

The report groups Levenshtein similarity values into descriptive bands. Those bands are useful for review and triage, but they are not universal validity thresholds. Projects should interpret them alongside the coding protocol, training process, adjudication rules, and the intended use of DCT outputs.

## Read Next

- `turns evaluate` usage guide: `docs/manual/04_modules/07_digital_conversational_turns/05_commands/02_evaluate/02_usage_guide.md`
- Digital Conversational Turns research context: `docs/manual/04_modules/07_digital_conversational_turns/03_research_context.md`
- Reliability outputs in generated examples: `docs/manual/03_features/04_generated_example_io.md`
