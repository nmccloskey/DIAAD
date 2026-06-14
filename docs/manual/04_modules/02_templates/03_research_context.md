# Templates Research Context

The Templates module exists because not every useful discourse-analysis workflow has a dedicated DIAAD analyzer. Some projects need a structured workbook for a local protocol, a reliability subset for a custom coding scheme, or a speaking-time table that feeds rates across modules.

In that sense, Templates is not an outcome module. It is a scaffolding module.

## Human-In-The-Loop Coding

Manual discourse coding depends on clear protocols, stable units, and reproducible file handling. Generic templates support those needs by giving coders a consistent table with identifiers, transcript-derived context, and optional coder assignments.

The module is especially useful when a research team has a project-specific coding paradigm. DIAAD can create the organized workbook, but the project remains responsible for defining the coding rules, training coders, checking reliability, and interpreting results.

## Sample Subsetting

`templates subset` is a general sample-selection utility. It was developed with reliability workflows in mind, especially for coding paradigms that DIAAD does not directly support, but it can also support pilot selection or other protocol-driven subsets.

The command tracks selected and excluded rows so reselection-style workflows can avoid already used samples when the input workbook marks them appropriately.

## Speaking Time

`templates times` creates a workbook for entering speaking-time values. DIAAD does not measure speaking time itself. Later rate commands use entered speaking-time values as denominators, typically converting seconds to minutes.

## Draft Review Notes

Before publication, review whether the module should give stronger guidance on reliability protocols for custom coding workflows, or whether that belongs entirely in functionality and workflow pages.
