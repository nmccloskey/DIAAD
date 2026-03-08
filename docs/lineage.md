# Project Lineage

The discourse-analysis tools in this ecosystem evolved through several stages
as the architecture and scope of the software matured.

## Early separation: monologic vs dialogic workflows

Initial development separated tools based on discourse structure.

### RASCAL
Focused on **monologic discourse analysis** and implemented the BU–TU
laboratory workflow, including:

- transcription reliability workflows
- complete utterance (CU) coding
- manual word counting
- CoreLex analysis

These components supported narrative and monologic discourse samples and were
organized around an Excel-based, revision-tolerant relational data structure.

### DIAAD (original version)

The original DIAAD project hosted modules designed for **dialogic discourse
analysis**, including:

- POWERS conversational coding
- digital conversation turn (DCT) analysis

These modules focused on clinician–client conversational data.

---

## Transition to TAALCR

During a later development phase, DIAAD was renamed **TAALCR**
(*Toolkit for Aggregate Analysis of Language in Conversation for Research*)
to support expanded conversational analysis functionality.

Planned additions included:

- flexible linguistic profile & dialogic alignment utilities for integration with external pipelines
  (e.g., ALASTR output)
- clinical language elicitation (CLE) modules

TAALCR emphasized SQLite-backed relational analysis with optional Excel
outputs for readability and navigation.

---

## Consolidation under DIAAD

As the architecture matured, the core design goal shifted toward a unified
database-oriented infrastructure for discourse analysis.

To reflect this shift, the RASCAL engine was generalized and renamed:

**DIAAD — Database-oriented, Integrative Architecture for Analyzing Discourse**

In this consolidated architecture:

DIAAD now serves as the **primary discourse analysis engine**, integrating
modules previously distributed across multiple repositories, including:

- transcription reliability workflows
- complete utterance coding
- manual word counting
- CoreLex analysis
- POWERS conversational coding
- digital conversation turn analysis

These modules operate within a revision-tolerant relational data model that
supports both automated and manual coding workflows.

---

## Current repository roles

### DIAAD
Primary engine for discourse analysis workflows and .xlsx-based relational data management.

### RASCAL
Maintained for historical continuity and may evolve into a laboratory-specific
wrapper providing preset workflows built on top of DIAAD.

### TAALCR
Being refocused as a conversational analysis engine centered on:

- clinical language elicitation (CLE)
- alignment-based conversational metrics
- integration with external linguistic analysis pipelines.

Future TAALCR releases will interact with DIAAD outputs rather than duplicating
core analysis modules.