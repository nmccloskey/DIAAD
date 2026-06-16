# Run Provenance and Audit Artifacts Research Context

DIAAD is designed for discourse-analysis workflows where outputs often pass through several stages: transcript preparation, reliability selection, manual coding, analysis, rate calculation, blinding, and export. In that setting, an output file is easier to interpret when the run that produced it is also preserved.

## Why Provenance Matters

Run provenance supports:

- reproducing analyses with the same settings;
- checking whether a result came from packaged defaults, project config, or CLI overrides;
- documenting random seeds for reliability and sampling procedures;
- identifying which input files were available at run time;
- distinguishing substantive outputs from logs and support files;
- diagnosing failed or partial runs;
- connecting manual coding stages to the command outputs that created their templates.

This is particularly important for research workflows that involve human review. If users create coding workbooks, edit them outside DIAAD, and later return them for analysis, the audit record should include both DIAAD run artifacts and the human-maintained file history.

## Configuration As Method

The effective configuration is a methodological artifact. It determines file discovery, identifier names, reliability sample fractions, transcript-processing settings, blinding settings, and module-specific filenames.

For reporting or internal review, it is usually not enough to say that a DIAAD command was run. The relevant configuration and any command-line overrides should be preserved with the run record.

## Limits

Provenance artifacts describe program execution. They do not replace:

- data governance records;
- institutional privacy procedures;
- manual coding protocols;
- version-controlled analysis scripts;
- human decisions made while revising transcripts or coding workbooks.

The artifacts also cannot prove that a workflow was methodologically appropriate. They make the workflow more inspectable.

## CLI And Web Records

CLI runs currently provide the clearest DIAAD audit trail because they write explicit provenance artifacts into the timestamped output directory. Web runs are useful for accessible execution and examples, but their ZIP output should be treated as a lighter operational record unless and until web provenance parity is implemented.

TODO: Review whether this CLI/web distinction should be strengthened for publication or updated after web provenance behavior changes.

## Read Next

- Configuration research context: `docs/manual/05_functionalities/01_configuration_sources_defaults_overrides/03_research_context.md`
- Blinding module: `docs/manual/04_modules/08_blinding/03_research_context.md`
- Transcript tabularization feature: `docs/manual/03_features/01_transcript_tabularization.md`
