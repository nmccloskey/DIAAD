# CLI Project Setup and First Run Research Context

The CLI workflow is not just a different interface. It supports a research-software style of working where configuration, inputs, outputs, and logs can be kept together and rerun.

## Reproducibility

Discourse-analysis workflows often span multiple days or personnel roles. A project may generate transcript tables, distribute manual coding files, evaluate reliability, revise coding rules, run analysis, and later calculate rates. CLI runs make those stages easier to preserve because outputs are written to stable local directories with run artifacts.

This does not make a workflow reproducible by itself. Users still need stable protocols, versioned configuration, preserved manually edited workbooks, and clear notes about changes between runs.

## Data Governance

Local CLI use is usually preferable when discourse data are identifiable, sensitive, large, or difficult to deidentify. This is especially relevant for dialogic samples where personal details may appear in transcript content even after filenames and sample IDs are masked.

Software blinding can help with coder-facing identifiers, but it cannot guarantee de-identification or practical masking. Sometimes formal blinding is technically easy but ineffective because coders know the participants, clinicians, or sessions. Treat blinding as one part of a project-specific governance plan.

## Automation And Human Review

The CLI makes it convenient to chain commands or script DIAAD inside a larger workflow. Use that power carefully. Commands that create manual coding workbooks should usually be followed by human completion, review, and reliability evaluation before downstream analysis.

## Read Next

- Run provenance research context: `docs/manual/05_functionalities/03_run_provenance_audit_artifacts/03_research_context.md`
- Blinding research context: `docs/manual/05_functionalities/14_blinding_unblinding_auto_blind/03_research_context.md`
- Reliability research context: `docs/manual/05_functionalities/12_reliability_selection_evaluation_reselection/03_research_context.md`
