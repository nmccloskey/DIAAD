# Blinding, Unblinding, and Auto-Blind Research Context

Blinding can reduce bias by limiting coder or analyst access to selected identifiers. In DIAAD, blinding is deterministic code replacement, not a full privacy system.

## Blinding Is Not De-Identification

A blinded workbook may hide `sample_id` while transcript content, comments, file context, or sample-specific events still reveal the participant or session. De-identification asks a broader question: whether the data contain information that could identify a person or sensitive context.

Blinding also may not achieve practical masking. For example, a team may transcribe clinician-client dialogs, deidentify file names, and blind sample identifiers for coding. If the same personnel who conducted the dialogs later code the data, their memory of the conversations may limit the practical value of formal blinding.

## Why Decode Before Analysis

Blinding is often most useful during human-facing coding. DIAAD analysis, by contrast, usually benefits from canonical identifiers because outputs need to reconnect with transcript tables, metadata, and other project files.

For that reason, decoding before DIAAD analysis is generally recommended after manual coding is complete. A later post-analysis encoding step can then support blinded statistical work or external sharing without forcing every DIAAD analysis command to operate on masked identifiers.

## Codebook Responsibility

The codebook is both a reproducibility artifact and a sensitive file. Losing it can make decoding difficult. Sharing it with blinded workbooks can defeat the masking. Store it deliberately, document which files it applies to, and avoid casual redistribution.

## Hosted And Local Workflows

Software behavior alone does not determine whether a workflow is appropriate for sensitive data. A hosted web workflow may be convenient for deidentified examples or low-risk materials. For identifiable or highly sensitive transcript content, a local CLI workflow may be more appropriate.

## Review Note

TODO: Before publication, review all privacy, de-identification, web-vs-local, and codebook-storage wording against the project's current deployment and data-governance expectations.

## Read Next

- Blinding module research context: `docs/manual/04_modules/08_blinding/03_research_context.md`
- Run provenance research context: `docs/manual/05_functionalities/03_run_provenance_audit_artifacts/03_research_context.md`
- Configuration research context: `docs/manual/05_functionalities/01_configuration_sources_defaults_overrides/03_research_context.md`
