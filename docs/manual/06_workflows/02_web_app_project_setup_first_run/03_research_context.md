# Web App Project Setup and First Run Research Context

The web app lowers the barrier to using DIAAD. It is useful when the main challenge is learning the file structure, trying a small project, or running a command without building a local scripting workflow.

## Accessibility And Control

A hosted app requires no installation, which can matter for itinerant work, teaching, quick checks, or users who cannot easily manage a Python environment. A local web app keeps the same interface while giving users more direct control over where processing occurs.

The CLI offers the most control for repeated research workflows, but not every user needs that control for every task.

## Privacy And Practical De-Identification

The web app processes runs in a temporary workspace and returns a downloadable ZIP. That design helps avoid retained session data, but it does not replace project-level privacy review.

Monologic narrative samples may be easy to deidentify well enough for hosted use. Clinician-client dialogs can be different: they may contain personal details, contextual clues, or content that is difficult to scrub without damaging the discourse sample. In those cases, local web or CLI use may be more appropriate even when the hosted service itself is designed to be temporary.

Formal blinding is also not always practically effective. If the same personnel collected, transcribed, and coded a conversation, masking the sample ID may not meaningfully hide the sample's identity.

## Workflow Staging

The web app is strongest when each stage has clear inputs and outputs. It is less natural for long automated pipelines because every stage requires upload, selection, execution, download, and inspection.

For manual coding workflows, this staged rhythm can be helpful. It forces users to pause at coding and reliability checkpoints before moving to analysis.

## Read Next

- CLI and web execution research context: `docs/manual/05_functionalities/02_cli_web_execution/`
- Blinding research context: `docs/manual/05_functionalities/14_blinding_unblinding_auto_blind/03_research_context.md`
- Run provenance: `docs/manual/05_functionalities/03_run_provenance_audit_artifacts/02_usage_guide.md`
