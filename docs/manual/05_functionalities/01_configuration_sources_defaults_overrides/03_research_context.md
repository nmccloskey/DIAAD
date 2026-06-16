# Configuration Sources, Defaults, and Overrides Research Context

In discourse-analysis software, configuration is not just a convenience layer. It is part of the methodological record. DIAAD settings determine which inputs are read, how samples are selected, how identifiers are interpreted, which files count as canonical inputs, and how optional processing choices are applied.

## Reproducibility

DIAAD's configuration model supports reproducibility by making project assumptions explicit. A run can be interpreted only in relation to the settings that produced it:

- `random_seed` affects reproducible sample selection and shuffling.
- `reliability_fraction` affects reliability sample size.
- `metadata_fields` affects extracted grouping variables.
- `sample_id_column` and `utterance_id_column` affect joins across transcript, coding, reliability, and analysis tables.
- filename settings affect which workbook is treated as the authoritative input.
- `auto_blind`, `blind_columns`, and `id_columns` affect masking and identifier recovery.
- transcript-processing settings such as `exclude_speakers`, `strip_clan`, `prefer_correction`, and `lowercase` affect some transcript-derived comparisons and outputs.

The effective configuration should therefore be treated as part of the analysis record, especially when results will be shared, audited, or reproduced later.

## Defaults And Method Claims

Packaged defaults make the program runnable, but defaults are not a substitute for project-specific methodological decisions. Some defaults are intentionally conservative. For example, `auto_tabularize` defaults to `false` so downstream commands do not silently create duplicate transcript-table representations or new sample IDs when the user expected to use an existing canonical transcript table.

Other settings, such as `exclude_speakers`, metadata fields, coding file names, and target vocabulary resources, are inherently project-specific. A published or shared analysis should describe the relevant settings rather than only saying that DIAAD was used.

## Overrides And Auditability

Command-line overrides are useful for temporary reruns, site-specific batches, and controlled sensitivity checks. They also increase the importance of preserving run artifacts, because the command line may no longer match the project YAML alone.

Normal CLI runs write the resolved configuration and override diff to the output logs. Dry-run config output can be saved before a run when the goal is to review or approve settings before processing data.

## CLI And Web Contexts

The CLI and web app use the same core configuration dataclasses after config files or web-built settings are normalized. They differ in context:

- CLI runs use user-controlled local paths.
- Web runs use temporary input and output folders for the session.
- The web config builder may present starter values that make the interface easier to understand, but those values should still be inspected as project settings.

For sensitive, identifiable, or hard-to-deidentify discourse data, configuration reproducibility should be paired with appropriate data governance. A resolved config records processing settings; it does not by itself deidentify transcript content or solve privacy risks.

## Read Next

- Configuration operation page: `docs/manual/02_operation/04_configuration.md`
- Run provenance and audit artifacts: `docs/manual/05_functionalities/03_run_provenance_audit_artifacts/03_research_context.md`
- Blinding module: `docs/manual/04_modules/08_blinding/01_quickstart.md`
- Transcript tabularization feature: `docs/manual/03_features/01_transcript_tabularization.md`
