# `turns analyze` Research Context

`turns analyze` is where DCT becomes more than a speaker tally. Because the coded string preserves order, the analysis can summarize both participation and sequential structure.

## Participation

Speaker, group, session, and bin summaries describe how much each speaker participated. These outputs are most interpretable when the project has stable speaker-code rules, stable bin definitions, and a clear decision about whether `0` represents only one clinician or a pooled non-client category.

## Transitions

Transition matrices estimate how often one speaker code is followed by another speaker code. Those matrices can be useful for questions about conversational dynamics, such as whether participant turns increasingly follow other participant turns rather than clinician prompts.

The transition-ratio sheet collapses those matrices into participant-to-participant, participant-to-clinician, and clinician-to-participant categories. That collapse is useful for overview, but it depends on the convention that `0` is the clinician or non-client category and all other digits are participant categories.

## Limits

DCT analysis is not lexical analysis. It does not inspect what speakers said, only the coded sequence of turns and optional dot markers.

The analysis also assumes that the conversation can be represented as a linear sequence. Overlap, simultaneous turns, or uncertain turn boundaries should be addressed in the project's coding protocol before analysis.

## Read Next

- `turns analyze` usage guide: `docs/manual/04_modules/07_digital_conversational_turns/05_commands/04_analyze/02_usage_guide.md`
- Digital Conversational Turns research context: `docs/manual/04_modules/07_digital_conversational_turns/03_research_context.md`
- Generated Example I/O: `docs/manual/03_features/04_generated_example_io.md`
