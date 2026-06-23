# Instruction Manual

**Version:** 0.0.0
**Generated:** 2026-06-16

---

## Manual Map (Tree)

```
├── 01_overview/
│   ├── 01_introduction.md — Introduction to DIAAD
│   ├── 02_manual_organization.md — Manual Organization
│   ├── 03_methodolgical_overview.md — Methodological Overview
│   └── 04_functional_overview.md — Functional Overview
├── 02_operation/
│   ├── 01_installation.md — Installation
│   ├── 02_command_line.md — Command-Line Operation
│   ├── 03_webapp.md — Web App Operation
│   ├── 04_configuration.md — Configuration
│   └── 05_testing.md — Testing
├── 03_features/
│   ├── 01_transcript_tabularization.md — Transcript Tabularization
│   ├── 02_word_counting_vs_target_vocabulary_coverage.md — Word Counting Versus Target Vocabulary Coverage
│   ├── 03_exact_file_name_matching.md — Exact File Name Matching
│   └── 04_generated_example_io.md — Generated Example I/O
├── 04_modules/
│   ├── 01_transcripts/
│   │   ├── 01_quickstart.md — Transcripts Module Quickstart
│   │   ├── 03_research_context.md — Transcripts Research Context
│   │   ├── 04_implementation_notes.md — Transcripts Implementation Notes
│   │   └── 05_commands/
│   │       ├── 01_tabularize/
│   │       │   ├── 01_quickstart.md — `transcripts tabularize` Quickstart
│   │       │   ├── 02_usage_guide.md — `transcripts tabularize` Usage Guide
│   │       │   ├── 03_research_context.md — `transcripts tabularize` Research Context
│   │       │   └── 04_implementation_notes.md — `transcripts tabularize` Implementation Notes
│   │       ├── 02_chats/
│   │       │   ├── 01_quickstart.md — `transcripts chats` Quickstart
│   │       │   ├── 02_usage_guide.md — `transcripts chats` Usage Guide
│   │       │   └── 04_implementation_notes.md — `transcripts chats` Implementation Notes
│   │       ├── 03_select/
│   │       │   ├── 01_quickstart.md — `transcripts select` Quickstart
│   │       │   ├── 02_usage_guide.md — `transcripts select` Usage Guide
│   │       │   └── 04_implementation_notes.md — `transcripts select` Implementation Notes
│   │       ├── 04_evaluate/
│   │       │   ├── 01_quickstart.md — `transcripts evaluate` Quickstart
│   │       │   ├── 02_usage_guide.md — `transcripts evaluate` Usage Guide
│   │       │   ├── 03_research_context.md — `transcripts evaluate` Research Context
│   │       │   └── 04_implementation_notes.md — `transcripts evaluate` Implementation Notes
│   │       └── 05_reselect/
│   │           ├── 01_quickstart.md — `transcripts reselect` Quickstart
│   │           ├── 02_usage_guide.md — `transcripts reselect` Usage Guide
│   │           └── 04_implementation_notes.md — `transcripts reselect` Implementation Notes
│   ├── 02_templates/
│   │   ├── 01_quickstart.md — Templates Module Quickstart
│   │   ├── 03_research_context.md — Templates Research Context
│   │   ├── 04_implementation_notes.md — Templates Implementation Notes
│   │   └── 05_commands/
│   │       ├── 01_utterances/
│   │       │   ├── 01_quickstart.md — `templates utterances` Quickstart
│   │       │   ├── 02_usage_guide.md — `templates utterances` Usage Guide
│   │       │   └── 04_implementation_notes.md — `templates utterances` Implementation Notes
│   │       ├── 02_samples/
│   │       │   ├── 01_quickstart.md — `templates samples` Quickstart
│   │       │   ├── 02_usage_guide.md — `templates samples` Usage Guide
│   │       │   └── 04_implementation_notes.md — `templates samples` Implementation Notes
│   │       ├── 03_times/
│   │       │   ├── 01_quickstart.md — `templates times` Quickstart
│   │       │   ├── 02_usage_guide.md — `templates times` Usage Guide
│   │       │   └── 04_implementation_notes.md — `templates times` Implementation Notes
│   │       ├── 04_subset/
│   │       │   ├── 01_quickstart.md — `templates subset` Quickstart
│   │       │   ├── 02_usage_guide.md — `templates subset` Usage Guide
│   │       │   └── 04_implementation_notes.md — `templates subset` Implementation Notes
│   │       └── 05_combine/
│   │           ├── 01_quickstart.md — `templates combine` Quickstart
│   │           ├── 02_usage_guide.md — `templates combine` Usage Guide
│   │           └── 04_implementation_notes.md — `templates combine` Implementation Notes
│   ├── 03_complete_utterances/
│   │   ├── 01_quickstart.md — Complete Utterances Module Quickstart
│   │   ├── 03_research_context.md — Complete Utterances Research Context
│   │   ├── 04_implementation_notes.md — Complete Utterances Implementation Notes
│   │   └── 05_commands/
│   │       ├── 01_files/
│   │       │   ├── 01_quickstart.md — `cus files` Quickstart
│   │       │   ├── 02_usage_guide.md — `cus files` Usage Guide
│   │       │   └── 04_implementation_notes.md — `cus files` Implementation Notes
│   │       ├── 02_evaluate/
│   │       │   ├── 01_quickstart.md — `cus evaluate` Quickstart
│   │       │   ├── 02_usage_guide.md — `cus evaluate` Usage Guide
│   │       │   ├── 03_research_context.md — `cus evaluate` Research Context
│   │       │   └── 04_implementation_notes.md — `cus evaluate` Implementation Notes
│   │       ├── 03_reselect/
│   │       │   ├── 01_quickstart.md — `cus reselect` Quickstart
│   │       │   ├── 02_usage_guide.md — `cus reselect` Usage Guide
│   │       │   └── 04_implementation_notes.md — `cus reselect` Implementation Notes
│   │       ├── 04_analyze/
│   │       │   ├── 01_quickstart.md — `cus analyze` Quickstart
│   │       │   ├── 02_usage_guide.md — `cus analyze` Usage Guide
│   │       │   └── 04_implementation_notes.md — `cus analyze` Implementation Notes
│   │       └── 05_rates/
│   │           ├── 01_quickstart.md — `cus rates` Quickstart
│   │           ├── 02_usage_guide.md — `cus rates` Usage Guide
│   │           └── 04_implementation_notes.md — `cus rates` Implementation Notes
│   ├── 04_word_counting/
│   │   ├── 01_quickstart.md — Word Counting Module Quickstart
│   │   ├── 03_research_context.md — Word Counting Research Context
│   │   ├── 04_implementation_notes.md — Word Counting Implementation Notes
│   │   └── 05_commands/
│   │       ├── 01_files/
│   │       │   ├── 01_quickstart.md — `words files` Quickstart
│   │       │   ├── 02_usage_guide.md — `words files` Usage Guide
│   │       │   └── 04_implementation_notes.md — `words files` Implementation Notes
│   │       ├── 02_evaluate/
│   │       │   ├── 01_quickstart.md — `words evaluate` Quickstart
│   │       │   ├── 02_usage_guide.md — `words evaluate` Usage Guide
│   │       │   ├── 03_research_context.md — `words evaluate` Research Context
│   │       │   └── 04_implementation_notes.md — `words evaluate` Implementation Notes
│   │       ├── 03_reselect/
│   │       │   ├── 01_quickstart.md — `words reselect` Quickstart
│   │       │   ├── 02_usage_guide.md — `words reselect` Usage Guide
│   │       │   └── 04_implementation_notes.md — `words reselect` Implementation Notes
│   │       ├── 04_analyze/
│   │       │   ├── 01_quickstart.md — `words analyze` Quickstart
│   │       │   ├── 02_usage_guide.md — `words analyze` Usage Guide
│   │       │   └── 04_implementation_notes.md — `words analyze` Implementation Notes
│   │       └── 05_rates/
│   │           ├── 01_quickstart.md — `words rates` Quickstart
│   │           ├── 02_usage_guide.md — `words rates` Usage Guide
│   │           └── 04_implementation_notes.md — `words rates` Implementation Notes
│   ├── 05_powers/
│   │   ├── 01_quickstart.md — POWERS Module Quickstart
│   │   ├── 03_research_context.md — POWERS Research Context
│   │   ├── 04_implementation_notes.md — POWERS Implementation Notes
│   │   └── 05_commands/
│   │       ├── 01_files/
│   │       │   ├── 01_quickstart.md — `powers files` Quickstart
│   │       │   ├── 02_usage_guide.md — `powers files` Usage Guide
│   │       │   └── 04_implementation_notes.md — `powers files` Implementation Notes
│   │       ├── 02_analyze/
│   │       │   ├── 01_quickstart.md — `powers analyze` Quickstart
│   │       │   ├── 02_usage_guide.md — `powers analyze` Usage Guide
│   │       │   └── 04_implementation_notes.md — `powers analyze` Implementation Notes
│   │       ├── 03_rates/
│   │       │   ├── 01_quickstart.md — `powers rates` Quickstart
│   │       │   ├── 02_usage_guide.md — `powers rates` Usage Guide
│   │       │   └── 04_implementation_notes.md — `powers rates` Implementation Notes
│   │       ├── 04_evaluate/
│   │       │   ├── 01_quickstart.md — `powers evaluate` Quickstart
│   │       │   ├── 02_usage_guide.md — `powers evaluate` Usage Guide
│   │       │   ├── 03_research_context.md — `powers evaluate` Research Context
│   │       │   └── 04_implementation_notes.md — `powers evaluate` Implementation Notes
│   │       └── 05_reselect/
│   │           ├── 01_quickstart.md — `powers reselect` Quickstart
│   │           ├── 02_usage_guide.md — `powers reselect` Usage Guide
│   │           └── 04_implementation_notes.md — `powers reselect` Implementation Notes
│   ├── 06_target_vocabulary_coverage/
│   │   ├── 01_quickstart.md — Target Vocabulary Coverage Module Quickstart
│   │   ├── 03_research_context.md — Target Vocabulary Coverage Research Context
│   │   ├── 04_implementation_notes.md — Target Vocabulary Coverage Implementation Notes
│   │   └── 05_commands/
│   │       ├── 01_file/
│   │       │   ├── 01_quickstart.md — `vocab file` Quickstart
│   │       │   ├── 02_usage_guide.md — `vocab file` Usage Guide
│   │       │   └── 04_implementation_notes.md — `vocab file` Implementation Notes
│   │       ├── 02_check/
│   │       │   ├── 01_quickstart.md — `vocab check` Quickstart
│   │       │   ├── 02_usage_guide.md — `vocab check` Usage Guide
│   │       │   └── 04_implementation_notes.md — `vocab check` Implementation Notes
│   │       ├── 03_analyze/
│   │       │   ├── 01_quickstart.md — `vocab analyze` Quickstart
│   │       │   ├── 02_usage_guide.md — `vocab analyze` Usage Guide
│   │       │   ├── 03_research_context.md — `vocab analyze` Research Context
│   │       │   └── 04_implementation_notes.md — `vocab analyze` Implementation Notes
│   │       └── 04_rates/
│   │           ├── 01_quickstart.md — `vocab rates` Quickstart
│   │           ├── 02_usage_guide.md — `vocab rates` Usage Guide
│   │           └── 04_implementation_notes.md — `vocab rates` Implementation Notes
│   ├── 07_digital_conversational_turns/
│   │   ├── 01_quickstart.md — Digital Conversational Turns Module Quickstart
│   │   ├── 03_research_context.md — Digital Conversational Turns Research Context
│   │   ├── 04_implementation_notes.md — Digital Conversational Turns Implementation Notes
│   │   └── 05_commands/
│   │       ├── 02_evaluate/
│   │       │   ├── 01_quickstart.md — `turns evaluate` Quickstart
│   │       │   ├── 02_usage_guide.md — `turns evaluate` Usage Guide
│   │       │   ├── 03_research_context.md — `turns evaluate` Research Context
│   │       │   └── 04_implementation_notes.md — `turns evaluate` Implementation Notes
│   │       └── 04_analyze/
│   │           ├── 01_quickstart.md — `turns analyze` Quickstart
│   │           ├── 02_usage_guide.md — `turns analyze` Usage Guide
│   │           ├── 03_research_context.md — `turns analyze` Research Context
│   │           └── 04_implementation_notes.md — `turns analyze` Implementation Notes
│   ├── 08_blinding/
│   │   ├── 01_quickstart.md — Blinding Module Quickstart
│   │   ├── 03_research_context.md — Blinding Research Context
│   │   ├── 04_implementation_notes.md — Blinding Implementation Notes
│   │   └── 05_commands/
│   │       ├── 01_encode/
│   │       │   ├── 01_quickstart.md — `blinding encode` Quickstart
│   │       │   ├── 02_usage_guide.md — `blinding encode` Usage Guide
│   │       │   └── 04_implementation_notes.md — `blinding encode` Implementation Notes
│   │       └── 02_decode/
│   │           ├── 01_quickstart.md — `blinding decode` Quickstart
│   │           ├── 02_usage_guide.md — `blinding decode` Usage Guide
│   │           └── 04_implementation_notes.md — `blinding decode` Implementation Notes
│   └── 09_examples/
│       ├── 01_quickstart.md — Examples Module Quickstart
│       ├── 02_usage_guide.md — Examples Module Usage Guide
│       ├── 03_research_context.md — Examples Research Context
│       ├── 04_implementation_notes.md — Examples Implementation Notes
│       └── 05_commands/
│           └── 01_examples/
│               ├── 01_quickstart.md — `examples` Quickstart
│               ├── 02_usage_guide.md — `examples` Usage Guide
│               └── 04_implementation_notes.md — `examples` Implementation Notes
├── 05_functionalities/
│   ├── 01_configuration_sources_defaults_overrides/
│   │   ├── 01_quickstart.md — Configuration Sources, Defaults, and Overrides Quickstart
│   │   ├── 02_usage_guide.md — Configuration Sources, Defaults, and Overrides Usage Guide
│   │   ├── 03_research_context.md — Configuration Sources, Defaults, and Overrides Research Context
│   │   └── 04_implementation_notes.md — Configuration Sources, Defaults, and Overrides Implementation Notes
│   ├── 02_cli_web_execution/
│   │   ├── 01_quickstart.md — CLI and Web Execution Quickstart
│   │   ├── 02_usage_guide.md — CLI and Web Execution Usage Guide
│   │   └── 04_implementation_notes.md — CLI and Web Execution Implementation Notes
│   ├── 03_run_provenance_audit_artifacts/
│   │   ├── 01_quickstart.md — Run Provenance and Audit Artifacts Quickstart
│   │   ├── 02_usage_guide.md — Run Provenance and Audit Artifacts Usage Guide
│   │   ├── 03_research_context.md — Run Provenance and Audit Artifacts Research Context
│   │   └── 04_implementation_notes.md — Run Provenance and Audit Artifacts Implementation Notes
│   ├── 04_example_package_generation_manifests/
│   │   ├── 01_quickstart.md — Example Package Generation and Manifests Quickstart
│   │   ├── 02_usage_guide.md — Example Package Generation and Manifests Usage Guide
│   │   └── 04_implementation_notes.md — Example Package Generation and Manifests Implementation Notes
│   ├── 05_generated_example_io_manual_composition/
│   │   ├── 01_quickstart.md — Generated Example I/O Manual Composition Quickstart
│   │   ├── 02_usage_guide.md — Generated Example I/O Manual Composition Usage Guide
│   │   └── 04_implementation_notes.md — Generated Example I/O Manual Composition Implementation Notes
│   ├── 06_transcript_preprocessing_tabularization_chat_export/
│   │   ├── 01_quickstart.md — Transcript Preprocessing, Tabularization, Auto-Tabularization, and CHAT Export Quickstart
│   │   ├── 02_usage_guide.md — Transcript Preprocessing, Tabularization, Auto-Tabularization, and CHAT Export Usage Guide
│   │   ├── 03_research_context.md — Transcript Preprocessing, Tabularization, Auto-Tabularization, and CHAT Export Research Context
│   │   └── 04_implementation_notes.md — Transcript Preprocessing, Tabularization, Auto-Tabularization, and CHAT Export Implementation Notes
│   ├── 07_transcript_text_normalization_speaker_exclusion/
│   │   ├── 01_quickstart.md — Transcript Text Normalization and Speaker Exclusion Quickstart
│   │   ├── 02_usage_guide.md — Transcript Text Normalization and Speaker Exclusion Usage Guide
│   │   ├── 03_research_context.md — Transcript Text Normalization and Speaker Exclusion Research Context
│   │   └── 04_implementation_notes.md — Transcript Text Normalization and Speaker Exclusion Implementation Notes
│   ├── 08_metadata_extraction/
│   │   ├── 01_quickstart.md — Metadata Extraction Quickstart
│   │   ├── 02_usage_guide.md — Metadata Extraction Usage Guide
│   │   ├── 03_research_context.md — Metadata Extraction Research Context
│   │   └── 04_implementation_notes.md — Metadata Extraction Implementation Notes
│   ├── 09_configured_filenames_file_discovery_input_selection/
│   │   ├── 01_quickstart.md — Configured Filenames, File Discovery, and Input Selection Quickstart
│   │   ├── 02_usage_guide.md — Configured Filenames, File Discovery, and Input Selection Usage Guide
│   │   └── 04_implementation_notes.md — Configured Filenames, File Discovery, and Input Selection Implementation Notes
│   ├── 10_configurable_sample_utterance_identifiers/
│   │   ├── 01_quickstart.md — Configurable Sample and Utterance Identifiers Quickstart
│   │   ├── 02_usage_guide.md — Configurable Sample and Utterance Identifiers Usage Guide
│   │   ├── 03_research_context.md — Configurable Sample and Utterance Identifiers Research Context
│   │   └── 04_implementation_notes.md — Configurable Sample and Utterance Identifiers Implementation Notes
│   ├── 11_revision_handling/
│   │   ├── 01_quickstart.md — Revision Handling Quickstart
│   │   ├── 02_usage_guide.md — Revision Handling Usage Guide
│   │   ├── 03_research_context.md — Revision Handling Research Context
│   │   └── 04_implementation_notes.md — Revision Handling Implementation Notes
│   ├── 12_reliability_selection_evaluation_reselection/
│   │   ├── 01_quickstart.md — Reliability Selection, Evaluation, and Reselection Quickstart
│   │   ├── 02_usage_guide.md — Reliability Selection, Evaluation, and Reselection Usage Guide
│   │   ├── 03_research_context.md — Reliability Selection, Evaluation, and Reselection Research Context
│   │   └── 04_implementation_notes.md — Reliability Selection, Evaluation, and Reselection Implementation Notes
│   ├── 13_sample_subsetting_resubsetting/
│   │   ├── 01_quickstart.md — General Sample Subsetting and Re-Subsetting Quickstart
│   │   ├── 02_usage_guide.md — General Sample Subsetting and Re-Subsetting Usage Guide
│   │   ├── 03_research_context.md — General Sample Subsetting and Re-Subsetting Research Context
│   │   └── 04_implementation_notes.md — General Sample Subsetting and Re-Subsetting Implementation Notes
│   ├── 14_blinding_unblinding_auto_blind/
│   │   ├── 01_quickstart.md — Blinding, Unblinding, and Auto-Blind Quickstart
│   │   ├── 02_usage_guide.md — Blinding, Unblinding, and Auto-Blind Usage Guide
│   │   ├── 03_research_context.md — Blinding, Unblinding, and Auto-Blind Research Context
│   │   └── 04_implementation_notes.md — Blinding, Unblinding, and Auto-Blind Implementation Notes
│   ├── 15_speaking_time_rate_calculation/
│   │   ├── 01_quickstart.md — Speaking-Time Rate Calculation Quickstart
│   │   ├── 02_usage_guide.md — Speaking-Time Rate Calculation Usage Guide
│   │   ├── 03_research_context.md — Speaking-Time Rate Calculation Research Context
│   │   └── 04_implementation_notes.md — Speaking-Time Rate Calculation Implementation Notes
│   ├── 16_target_vocabulary_resource_management/
│   │   ├── 01_quickstart.md — Target Vocabulary Resource Management Quickstart
│   │   ├── 02_usage_guide.md — Target Vocabulary Resource Management Usage Guide
│   │   ├── 03_research_context.md — Target Vocabulary Resource Management Research Context
│   │   └── 04_implementation_notes.md — Target Vocabulary Resource Management Implementation Notes
│   ├── 17_powers_automation_support/
│   │   ├── 01_quickstart.md — POWERS Automation Support Quickstart
│   │   ├── 02_usage_guide.md — POWERS Automation Support Usage Guide
│   │   ├── 03_research_context.md — POWERS Automation Support Research Context
│   │   └── 04_implementation_notes.md — POWERS Automation Support Implementation Notes
│   └── 18_coder_identifier_columns/
│       ├── 01_quickstart.md — Coder Identifier Columns Quickstart
│       ├── 02_usage_guide.md — Coder Identifier Columns Usage Guide
│       ├── 03_research_context.md — Coder Identifier Columns Research Context
│       └── 04_implementation_notes.md — Coder Identifier Columns Implementation Notes
├── 06_workflows/
│   ├── 01_cli_project_setup_first_run/
│   │   ├── 01_quickstart.md — CLI Project Setup and First Run Quickstart
│   │   ├── 02_usage_guide.md — CLI Project Setup and First Run Usage Guide
│   │   ├── 03_research_context.md — CLI Project Setup and First Run Research Context
│   │   └── 04_implementation_notes.md — CLI Project Setup and First Run Implementation Notes
│   ├── 02_web_app_project_setup_first_run/
│   │   ├── 01_quickstart.md — Web App Project Setup and First Run Quickstart
│   │   ├── 02_usage_guide.md — Web App Project Setup and First Run Usage Guide
│   │   ├── 03_research_context.md — Web App Project Setup and First Run Research Context
│   │   └── 04_implementation_notes.md — Web App Project Setup and First Run Implementation Notes
│   ├── 03_example_dataset_command_specific_packages/
│   │   ├── 01_quickstart.md — Example Dataset and Command-Specific Packages Quickstart
│   │   ├── 02_usage_guide.md — Example Dataset and Command-Specific Packages Usage Guide
│   │   ├── 03_research_context.md — Example Dataset and Command-Specific Packages Research Context
│   │   └── 04_implementation_notes.md — Example Dataset and Command-Specific Packages Implementation Notes
│   ├── 04_transcription_based_workflow_baseline/
│   │   ├── 01_quickstart.md — Transcription-Based Workflow Baseline Quickstart
│   │   ├── 02_usage_guide.md — Transcription-Based Workflow Baseline Usage Guide
│   │   ├── 03_research_context.md — Transcription-Based Workflow Baseline Research Context
│   │   └── 04_implementation_notes.md — Transcription-Based Workflow Baseline Implementation Notes
│   ├── 05_transcription_reliability/
│   │   ├── 01_quickstart.md — Transcription Reliability Quickstart
│   │   ├── 02_usage_guide.md — Transcription Reliability Usage Guide
│   │   └── 03_research_context.md — Transcription Reliability Research Context
│   ├── 06_transcript_table_revision_chat_export/
│   │   ├── 01_quickstart.md — Transcript Table Revision and CHAT Export Quickstart
│   │   ├── 02_usage_guide.md — Transcript Table Revision and CHAT Export Usage Guide
│   │   └── 03_research_context.md — Transcript Table Revision and CHAT Export Research Context
│   ├── 07_monologic_narrative_complete_utterances/
│   │   ├── 01_quickstart.md — Monologic Narrative Complete Utterances Quickstart
│   │   ├── 02_usage_guide.md — Monologic Narrative Complete Utterances Usage Guide
│   │   └── 03_research_context.md — Monologic Narrative Complete Utterances Research Context
│   ├── 08_monologic_narrative_word_counting/
│   │   ├── 01_quickstart.md — Monologic Narrative Word Counting Quickstart
│   │   ├── 02_usage_guide.md — Monologic Narrative Word Counting Usage Guide
│   │   └── 03_research_context.md — Monologic Narrative Word Counting Research Context
│   ├── 09_monologic_narrative_target_vocabulary_coverage/
│   │   ├── 01_quickstart.md — Monologic Narrative Target Vocabulary Coverage Quickstart
│   │   ├── 02_usage_guide.md — Monologic Narrative Target Vocabulary Coverage Usage Guide
│   │   └── 03_research_context.md — Monologic Narrative Target Vocabulary Coverage Research Context
│   ├── 10_digital_conversational_turns/
│   │   ├── 01_quickstart.md — Digital Conversational Turns Quickstart
│   │   ├── 02_usage_guide.md — Digital Conversational Turns Usage Guide
│   │   └── 03_research_context.md — Digital Conversational Turns Research Context
│   ├── 11_monologic_narrative_integrated_workflow/
│   │   ├── 01_quickstart.md — Monologic Narrative Integrated Workflow Quickstart
│   │   ├── 02_usage_guide.md — Monologic Narrative Integrated Workflow Usage Guide
│   │   └── 03_research_context.md — Monologic Narrative Integrated Workflow Research Context
│   └── 12_clinician_client_dialogue_powers/
│       ├── 01_quickstart.md — Clinician-Client Dialogue POWERS Quickstart
│       ├── 02_usage_guide.md — Clinician-Client Dialogue POWERS Usage Guide
│       └── 03_research_context.md — Clinician-Client Dialogue POWERS Research Context
└── 99_references.md — References
```

## Outline (Links)

### 01_overview/
- [01_introduction.md — Introduction to DIAAD](01_overview/01_introduction.md)
- [02_manual_organization.md — Manual Organization](01_overview/02_manual_organization.md)
- [03_methodolgical_overview.md — Methodological Overview](01_overview/03_methodolgical_overview.md)
- [04_functional_overview.md — Functional Overview](01_overview/04_functional_overview.md)

### 02_operation/
- [01_installation.md — Installation](02_operation/01_installation.md)
- [02_command_line.md — Command-Line Operation](02_operation/02_command_line.md)
- [03_webapp.md — Web App Operation](02_operation/03_webapp.md)
- [04_configuration.md — Configuration](02_operation/04_configuration.md)
- [05_testing.md — Testing](02_operation/05_testing.md)

### 03_features/
- [01_transcript_tabularization.md — Transcript Tabularization](03_features/01_transcript_tabularization.md)
- [02_word_counting_vs_target_vocabulary_coverage.md — Word Counting Versus Target Vocabulary Coverage](03_features/02_word_counting_vs_target_vocabulary_coverage.md)
- [03_exact_file_name_matching.md — Exact File Name Matching](03_features/03_exact_file_name_matching.md)
- [04_generated_example_io.md — Generated Example I/O](03_features/04_generated_example_io.md)

### 04_modules/01_transcripts/
- [01_quickstart.md — Transcripts Module Quickstart](04_modules/01_transcripts/01_quickstart.md)
- [03_research_context.md — Transcripts Research Context](04_modules/01_transcripts/03_research_context.md)
- [04_implementation_notes.md — Transcripts Implementation Notes](04_modules/01_transcripts/04_implementation_notes.md)

### 04_modules/01_transcripts/05_commands/01_tabularize/
- [01_quickstart.md — `transcripts tabularize` Quickstart](04_modules/01_transcripts/05_commands/01_tabularize/01_quickstart.md)
- [02_usage_guide.md — `transcripts tabularize` Usage Guide](04_modules/01_transcripts/05_commands/01_tabularize/02_usage_guide.md)
- [03_research_context.md — `transcripts tabularize` Research Context](04_modules/01_transcripts/05_commands/01_tabularize/03_research_context.md)
- [04_implementation_notes.md — `transcripts tabularize` Implementation Notes](04_modules/01_transcripts/05_commands/01_tabularize/04_implementation_notes.md)

### 04_modules/01_transcripts/05_commands/02_chats/
- [01_quickstart.md — `transcripts chats` Quickstart](04_modules/01_transcripts/05_commands/02_chats/01_quickstart.md)
- [02_usage_guide.md — `transcripts chats` Usage Guide](04_modules/01_transcripts/05_commands/02_chats/02_usage_guide.md)
- [04_implementation_notes.md — `transcripts chats` Implementation Notes](04_modules/01_transcripts/05_commands/02_chats/04_implementation_notes.md)

### 04_modules/01_transcripts/05_commands/03_select/
- [01_quickstart.md — `transcripts select` Quickstart](04_modules/01_transcripts/05_commands/03_select/01_quickstart.md)
- [02_usage_guide.md — `transcripts select` Usage Guide](04_modules/01_transcripts/05_commands/03_select/02_usage_guide.md)
- [04_implementation_notes.md — `transcripts select` Implementation Notes](04_modules/01_transcripts/05_commands/03_select/04_implementation_notes.md)

### 04_modules/01_transcripts/05_commands/04_evaluate/
- [01_quickstart.md — `transcripts evaluate` Quickstart](04_modules/01_transcripts/05_commands/04_evaluate/01_quickstart.md)
- [02_usage_guide.md — `transcripts evaluate` Usage Guide](04_modules/01_transcripts/05_commands/04_evaluate/02_usage_guide.md)
- [03_research_context.md — `transcripts evaluate` Research Context](04_modules/01_transcripts/05_commands/04_evaluate/03_research_context.md)
- [04_implementation_notes.md — `transcripts evaluate` Implementation Notes](04_modules/01_transcripts/05_commands/04_evaluate/04_implementation_notes.md)

### 04_modules/01_transcripts/05_commands/05_reselect/
- [01_quickstart.md — `transcripts reselect` Quickstart](04_modules/01_transcripts/05_commands/05_reselect/01_quickstart.md)
- [02_usage_guide.md — `transcripts reselect` Usage Guide](04_modules/01_transcripts/05_commands/05_reselect/02_usage_guide.md)
- [04_implementation_notes.md — `transcripts reselect` Implementation Notes](04_modules/01_transcripts/05_commands/05_reselect/04_implementation_notes.md)

### 04_modules/02_templates/
- [01_quickstart.md — Templates Module Quickstart](04_modules/02_templates/01_quickstart.md)
- [03_research_context.md — Templates Research Context](04_modules/02_templates/03_research_context.md)
- [04_implementation_notes.md — Templates Implementation Notes](04_modules/02_templates/04_implementation_notes.md)

### 04_modules/02_templates/05_commands/01_utterances/
- [01_quickstart.md — `templates utterances` Quickstart](04_modules/02_templates/05_commands/01_utterances/01_quickstart.md)
- [02_usage_guide.md — `templates utterances` Usage Guide](04_modules/02_templates/05_commands/01_utterances/02_usage_guide.md)
- [04_implementation_notes.md — `templates utterances` Implementation Notes](04_modules/02_templates/05_commands/01_utterances/04_implementation_notes.md)

### 04_modules/02_templates/05_commands/02_samples/
- [01_quickstart.md — `templates samples` Quickstart](04_modules/02_templates/05_commands/02_samples/01_quickstart.md)
- [02_usage_guide.md — `templates samples` Usage Guide](04_modules/02_templates/05_commands/02_samples/02_usage_guide.md)
- [04_implementation_notes.md — `templates samples` Implementation Notes](04_modules/02_templates/05_commands/02_samples/04_implementation_notes.md)

### 04_modules/02_templates/05_commands/03_times/
- [01_quickstart.md — `templates times` Quickstart](04_modules/02_templates/05_commands/03_times/01_quickstart.md)
- [02_usage_guide.md — `templates times` Usage Guide](04_modules/02_templates/05_commands/03_times/02_usage_guide.md)
- [04_implementation_notes.md — `templates times` Implementation Notes](04_modules/02_templates/05_commands/03_times/04_implementation_notes.md)

### 04_modules/02_templates/05_commands/04_subset/
- [01_quickstart.md — `templates subset` Quickstart](04_modules/02_templates/05_commands/04_subset/01_quickstart.md)
- [02_usage_guide.md — `templates subset` Usage Guide](04_modules/02_templates/05_commands/04_subset/02_usage_guide.md)
- [04_implementation_notes.md — `templates subset` Implementation Notes](04_modules/02_templates/05_commands/04_subset/04_implementation_notes.md)

### 04_modules/02_templates/05_commands/05_combine/
- [01_quickstart.md — `templates combine` Quickstart](04_modules/02_templates/05_commands/05_combine/01_quickstart.md)
- [02_usage_guide.md — `templates combine` Usage Guide](04_modules/02_templates/05_commands/05_combine/02_usage_guide.md)
- [04_implementation_notes.md — `templates combine` Implementation Notes](04_modules/02_templates/05_commands/05_combine/04_implementation_notes.md)

### 04_modules/03_complete_utterances/
- [01_quickstart.md — Complete Utterances Module Quickstart](04_modules/03_complete_utterances/01_quickstart.md)
- [03_research_context.md — Complete Utterances Research Context](04_modules/03_complete_utterances/03_research_context.md)
- [04_implementation_notes.md — Complete Utterances Implementation Notes](04_modules/03_complete_utterances/04_implementation_notes.md)

### 04_modules/03_complete_utterances/05_commands/01_files/
- [01_quickstart.md — `cus files` Quickstart](04_modules/03_complete_utterances/05_commands/01_files/01_quickstart.md)
- [02_usage_guide.md — `cus files` Usage Guide](04_modules/03_complete_utterances/05_commands/01_files/02_usage_guide.md)
- [04_implementation_notes.md — `cus files` Implementation Notes](04_modules/03_complete_utterances/05_commands/01_files/04_implementation_notes.md)

### 04_modules/03_complete_utterances/05_commands/02_evaluate/
- [01_quickstart.md — `cus evaluate` Quickstart](04_modules/03_complete_utterances/05_commands/02_evaluate/01_quickstart.md)
- [02_usage_guide.md — `cus evaluate` Usage Guide](04_modules/03_complete_utterances/05_commands/02_evaluate/02_usage_guide.md)
- [03_research_context.md — `cus evaluate` Research Context](04_modules/03_complete_utterances/05_commands/02_evaluate/03_research_context.md)
- [04_implementation_notes.md — `cus evaluate` Implementation Notes](04_modules/03_complete_utterances/05_commands/02_evaluate/04_implementation_notes.md)

### 04_modules/03_complete_utterances/05_commands/03_reselect/
- [01_quickstart.md — `cus reselect` Quickstart](04_modules/03_complete_utterances/05_commands/03_reselect/01_quickstart.md)
- [02_usage_guide.md — `cus reselect` Usage Guide](04_modules/03_complete_utterances/05_commands/03_reselect/02_usage_guide.md)
- [04_implementation_notes.md — `cus reselect` Implementation Notes](04_modules/03_complete_utterances/05_commands/03_reselect/04_implementation_notes.md)

### 04_modules/03_complete_utterances/05_commands/04_analyze/
- [01_quickstart.md — `cus analyze` Quickstart](04_modules/03_complete_utterances/05_commands/04_analyze/01_quickstart.md)
- [02_usage_guide.md — `cus analyze` Usage Guide](04_modules/03_complete_utterances/05_commands/04_analyze/02_usage_guide.md)
- [04_implementation_notes.md — `cus analyze` Implementation Notes](04_modules/03_complete_utterances/05_commands/04_analyze/04_implementation_notes.md)

### 04_modules/03_complete_utterances/05_commands/05_rates/
- [01_quickstart.md — `cus rates` Quickstart](04_modules/03_complete_utterances/05_commands/05_rates/01_quickstart.md)
- [02_usage_guide.md — `cus rates` Usage Guide](04_modules/03_complete_utterances/05_commands/05_rates/02_usage_guide.md)
- [04_implementation_notes.md — `cus rates` Implementation Notes](04_modules/03_complete_utterances/05_commands/05_rates/04_implementation_notes.md)

### 04_modules/04_word_counting/
- [01_quickstart.md — Word Counting Module Quickstart](04_modules/04_word_counting/01_quickstart.md)
- [03_research_context.md — Word Counting Research Context](04_modules/04_word_counting/03_research_context.md)
- [04_implementation_notes.md — Word Counting Implementation Notes](04_modules/04_word_counting/04_implementation_notes.md)

### 04_modules/04_word_counting/05_commands/01_files/
- [01_quickstart.md — `words files` Quickstart](04_modules/04_word_counting/05_commands/01_files/01_quickstart.md)
- [02_usage_guide.md — `words files` Usage Guide](04_modules/04_word_counting/05_commands/01_files/02_usage_guide.md)
- [04_implementation_notes.md — `words files` Implementation Notes](04_modules/04_word_counting/05_commands/01_files/04_implementation_notes.md)

### 04_modules/04_word_counting/05_commands/02_evaluate/
- [01_quickstart.md — `words evaluate` Quickstart](04_modules/04_word_counting/05_commands/02_evaluate/01_quickstart.md)
- [02_usage_guide.md — `words evaluate` Usage Guide](04_modules/04_word_counting/05_commands/02_evaluate/02_usage_guide.md)
- [03_research_context.md — `words evaluate` Research Context](04_modules/04_word_counting/05_commands/02_evaluate/03_research_context.md)
- [04_implementation_notes.md — `words evaluate` Implementation Notes](04_modules/04_word_counting/05_commands/02_evaluate/04_implementation_notes.md)

### 04_modules/04_word_counting/05_commands/03_reselect/
- [01_quickstart.md — `words reselect` Quickstart](04_modules/04_word_counting/05_commands/03_reselect/01_quickstart.md)
- [02_usage_guide.md — `words reselect` Usage Guide](04_modules/04_word_counting/05_commands/03_reselect/02_usage_guide.md)
- [04_implementation_notes.md — `words reselect` Implementation Notes](04_modules/04_word_counting/05_commands/03_reselect/04_implementation_notes.md)

### 04_modules/04_word_counting/05_commands/04_analyze/
- [01_quickstart.md — `words analyze` Quickstart](04_modules/04_word_counting/05_commands/04_analyze/01_quickstart.md)
- [02_usage_guide.md — `words analyze` Usage Guide](04_modules/04_word_counting/05_commands/04_analyze/02_usage_guide.md)
- [04_implementation_notes.md — `words analyze` Implementation Notes](04_modules/04_word_counting/05_commands/04_analyze/04_implementation_notes.md)

### 04_modules/04_word_counting/05_commands/05_rates/
- [01_quickstart.md — `words rates` Quickstart](04_modules/04_word_counting/05_commands/05_rates/01_quickstart.md)
- [02_usage_guide.md — `words rates` Usage Guide](04_modules/04_word_counting/05_commands/05_rates/02_usage_guide.md)
- [04_implementation_notes.md — `words rates` Implementation Notes](04_modules/04_word_counting/05_commands/05_rates/04_implementation_notes.md)

### 04_modules/05_powers/
- [01_quickstart.md — POWERS Module Quickstart](04_modules/05_powers/01_quickstart.md)
- [03_research_context.md — POWERS Research Context](04_modules/05_powers/03_research_context.md)
- [04_implementation_notes.md — POWERS Implementation Notes](04_modules/05_powers/04_implementation_notes.md)

### 04_modules/05_powers/05_commands/01_files/
- [01_quickstart.md — `powers files` Quickstart](04_modules/05_powers/05_commands/01_files/01_quickstart.md)
- [02_usage_guide.md — `powers files` Usage Guide](04_modules/05_powers/05_commands/01_files/02_usage_guide.md)
- [04_implementation_notes.md — `powers files` Implementation Notes](04_modules/05_powers/05_commands/01_files/04_implementation_notes.md)

### 04_modules/05_powers/05_commands/02_analyze/
- [01_quickstart.md — `powers analyze` Quickstart](04_modules/05_powers/05_commands/02_analyze/01_quickstart.md)
- [02_usage_guide.md — `powers analyze` Usage Guide](04_modules/05_powers/05_commands/02_analyze/02_usage_guide.md)
- [04_implementation_notes.md — `powers analyze` Implementation Notes](04_modules/05_powers/05_commands/02_analyze/04_implementation_notes.md)

### 04_modules/05_powers/05_commands/03_rates/
- [01_quickstart.md — `powers rates` Quickstart](04_modules/05_powers/05_commands/03_rates/01_quickstart.md)
- [02_usage_guide.md — `powers rates` Usage Guide](04_modules/05_powers/05_commands/03_rates/02_usage_guide.md)
- [04_implementation_notes.md — `powers rates` Implementation Notes](04_modules/05_powers/05_commands/03_rates/04_implementation_notes.md)

### 04_modules/05_powers/05_commands/04_evaluate/
- [01_quickstart.md — `powers evaluate` Quickstart](04_modules/05_powers/05_commands/04_evaluate/01_quickstart.md)
- [02_usage_guide.md — `powers evaluate` Usage Guide](04_modules/05_powers/05_commands/04_evaluate/02_usage_guide.md)
- [03_research_context.md — `powers evaluate` Research Context](04_modules/05_powers/05_commands/04_evaluate/03_research_context.md)
- [04_implementation_notes.md — `powers evaluate` Implementation Notes](04_modules/05_powers/05_commands/04_evaluate/04_implementation_notes.md)

### 04_modules/05_powers/05_commands/05_reselect/
- [01_quickstart.md — `powers reselect` Quickstart](04_modules/05_powers/05_commands/05_reselect/01_quickstart.md)
- [02_usage_guide.md — `powers reselect` Usage Guide](04_modules/05_powers/05_commands/05_reselect/02_usage_guide.md)
- [04_implementation_notes.md — `powers reselect` Implementation Notes](04_modules/05_powers/05_commands/05_reselect/04_implementation_notes.md)

### 04_modules/06_target_vocabulary_coverage/
- [01_quickstart.md — Target Vocabulary Coverage Module Quickstart](04_modules/06_target_vocabulary_coverage/01_quickstart.md)
- [03_research_context.md — Target Vocabulary Coverage Research Context](04_modules/06_target_vocabulary_coverage/03_research_context.md)
- [04_implementation_notes.md — Target Vocabulary Coverage Implementation Notes](04_modules/06_target_vocabulary_coverage/04_implementation_notes.md)

### 04_modules/06_target_vocabulary_coverage/05_commands/01_file/
- [01_quickstart.md — `vocab file` Quickstart](04_modules/06_target_vocabulary_coverage/05_commands/01_file/01_quickstart.md)
- [02_usage_guide.md — `vocab file` Usage Guide](04_modules/06_target_vocabulary_coverage/05_commands/01_file/02_usage_guide.md)
- [04_implementation_notes.md — `vocab file` Implementation Notes](04_modules/06_target_vocabulary_coverage/05_commands/01_file/04_implementation_notes.md)

### 04_modules/06_target_vocabulary_coverage/05_commands/02_check/
- [01_quickstart.md — `vocab check` Quickstart](04_modules/06_target_vocabulary_coverage/05_commands/02_check/01_quickstart.md)
- [02_usage_guide.md — `vocab check` Usage Guide](04_modules/06_target_vocabulary_coverage/05_commands/02_check/02_usage_guide.md)
- [04_implementation_notes.md — `vocab check` Implementation Notes](04_modules/06_target_vocabulary_coverage/05_commands/02_check/04_implementation_notes.md)

### 04_modules/06_target_vocabulary_coverage/05_commands/03_analyze/
- [01_quickstart.md — `vocab analyze` Quickstart](04_modules/06_target_vocabulary_coverage/05_commands/03_analyze/01_quickstart.md)
- [02_usage_guide.md — `vocab analyze` Usage Guide](04_modules/06_target_vocabulary_coverage/05_commands/03_analyze/02_usage_guide.md)
- [03_research_context.md — `vocab analyze` Research Context](04_modules/06_target_vocabulary_coverage/05_commands/03_analyze/03_research_context.md)
- [04_implementation_notes.md — `vocab analyze` Implementation Notes](04_modules/06_target_vocabulary_coverage/05_commands/03_analyze/04_implementation_notes.md)

### 04_modules/06_target_vocabulary_coverage/05_commands/04_rates/
- [01_quickstart.md — `vocab rates` Quickstart](04_modules/06_target_vocabulary_coverage/05_commands/04_rates/01_quickstart.md)
- [02_usage_guide.md — `vocab rates` Usage Guide](04_modules/06_target_vocabulary_coverage/05_commands/04_rates/02_usage_guide.md)
- [04_implementation_notes.md — `vocab rates` Implementation Notes](04_modules/06_target_vocabulary_coverage/05_commands/04_rates/04_implementation_notes.md)

### 04_modules/07_digital_conversational_turns/
- [01_quickstart.md — Digital Conversational Turns Module Quickstart](04_modules/07_digital_conversational_turns/01_quickstart.md)
- [03_research_context.md — Digital Conversational Turns Research Context](04_modules/07_digital_conversational_turns/03_research_context.md)
- [04_implementation_notes.md — Digital Conversational Turns Implementation Notes](04_modules/07_digital_conversational_turns/04_implementation_notes.md)

### 04_modules/07_digital_conversational_turns/05_commands/02_evaluate/
- [01_quickstart.md — `turns evaluate` Quickstart](04_modules/07_digital_conversational_turns/05_commands/02_evaluate/01_quickstart.md)
- [02_usage_guide.md — `turns evaluate` Usage Guide](04_modules/07_digital_conversational_turns/05_commands/02_evaluate/02_usage_guide.md)
- [03_research_context.md — `turns evaluate` Research Context](04_modules/07_digital_conversational_turns/05_commands/02_evaluate/03_research_context.md)
- [04_implementation_notes.md — `turns evaluate` Implementation Notes](04_modules/07_digital_conversational_turns/05_commands/02_evaluate/04_implementation_notes.md)

### 04_modules/07_digital_conversational_turns/05_commands/04_analyze/
- [01_quickstart.md — `turns analyze` Quickstart](04_modules/07_digital_conversational_turns/05_commands/04_analyze/01_quickstart.md)
- [02_usage_guide.md — `turns analyze` Usage Guide](04_modules/07_digital_conversational_turns/05_commands/04_analyze/02_usage_guide.md)
- [03_research_context.md — `turns analyze` Research Context](04_modules/07_digital_conversational_turns/05_commands/04_analyze/03_research_context.md)
- [04_implementation_notes.md — `turns analyze` Implementation Notes](04_modules/07_digital_conversational_turns/05_commands/04_analyze/04_implementation_notes.md)

### 04_modules/08_blinding/
- [01_quickstart.md — Blinding Module Quickstart](04_modules/08_blinding/01_quickstart.md)
- [03_research_context.md — Blinding Research Context](04_modules/08_blinding/03_research_context.md)
- [04_implementation_notes.md — Blinding Implementation Notes](04_modules/08_blinding/04_implementation_notes.md)

### 04_modules/08_blinding/05_commands/01_encode/
- [01_quickstart.md — `blinding encode` Quickstart](04_modules/08_blinding/05_commands/01_encode/01_quickstart.md)
- [02_usage_guide.md — `blinding encode` Usage Guide](04_modules/08_blinding/05_commands/01_encode/02_usage_guide.md)
- [04_implementation_notes.md — `blinding encode` Implementation Notes](04_modules/08_blinding/05_commands/01_encode/04_implementation_notes.md)

### 04_modules/08_blinding/05_commands/02_decode/
- [01_quickstart.md — `blinding decode` Quickstart](04_modules/08_blinding/05_commands/02_decode/01_quickstart.md)
- [02_usage_guide.md — `blinding decode` Usage Guide](04_modules/08_blinding/05_commands/02_decode/02_usage_guide.md)
- [04_implementation_notes.md — `blinding decode` Implementation Notes](04_modules/08_blinding/05_commands/02_decode/04_implementation_notes.md)

### 04_modules/09_examples/
- [01_quickstart.md — Examples Module Quickstart](04_modules/09_examples/01_quickstart.md)
- [02_usage_guide.md — Examples Module Usage Guide](04_modules/09_examples/02_usage_guide.md)
- [03_research_context.md — Examples Research Context](04_modules/09_examples/03_research_context.md)
- [04_implementation_notes.md — Examples Implementation Notes](04_modules/09_examples/04_implementation_notes.md)

### 04_modules/09_examples/05_commands/01_examples/
- [01_quickstart.md — `examples` Quickstart](04_modules/09_examples/05_commands/01_examples/01_quickstart.md)
- [02_usage_guide.md — `examples` Usage Guide](04_modules/09_examples/05_commands/01_examples/02_usage_guide.md)
- [04_implementation_notes.md — `examples` Implementation Notes](04_modules/09_examples/05_commands/01_examples/04_implementation_notes.md)

### 05_functionalities/01_configuration_sources_defaults_overrides/
- [01_quickstart.md — Configuration Sources, Defaults, and Overrides Quickstart](05_functionalities/01_configuration_sources_defaults_overrides/01_quickstart.md)
- [02_usage_guide.md — Configuration Sources, Defaults, and Overrides Usage Guide](05_functionalities/01_configuration_sources_defaults_overrides/02_usage_guide.md)
- [03_research_context.md — Configuration Sources, Defaults, and Overrides Research Context](05_functionalities/01_configuration_sources_defaults_overrides/03_research_context.md)
- [04_implementation_notes.md — Configuration Sources, Defaults, and Overrides Implementation Notes](05_functionalities/01_configuration_sources_defaults_overrides/04_implementation_notes.md)

### 05_functionalities/02_cli_web_execution/
- [01_quickstart.md — CLI and Web Execution Quickstart](05_functionalities/02_cli_web_execution/01_quickstart.md)
- [02_usage_guide.md — CLI and Web Execution Usage Guide](05_functionalities/02_cli_web_execution/02_usage_guide.md)
- [04_implementation_notes.md — CLI and Web Execution Implementation Notes](05_functionalities/02_cli_web_execution/04_implementation_notes.md)

### 05_functionalities/03_run_provenance_audit_artifacts/
- [01_quickstart.md — Run Provenance and Audit Artifacts Quickstart](05_functionalities/03_run_provenance_audit_artifacts/01_quickstart.md)
- [02_usage_guide.md — Run Provenance and Audit Artifacts Usage Guide](05_functionalities/03_run_provenance_audit_artifacts/02_usage_guide.md)
- [03_research_context.md — Run Provenance and Audit Artifacts Research Context](05_functionalities/03_run_provenance_audit_artifacts/03_research_context.md)
- [04_implementation_notes.md — Run Provenance and Audit Artifacts Implementation Notes](05_functionalities/03_run_provenance_audit_artifacts/04_implementation_notes.md)

### 05_functionalities/04_example_package_generation_manifests/
- [01_quickstart.md — Example Package Generation and Manifests Quickstart](05_functionalities/04_example_package_generation_manifests/01_quickstart.md)
- [02_usage_guide.md — Example Package Generation and Manifests Usage Guide](05_functionalities/04_example_package_generation_manifests/02_usage_guide.md)
- [04_implementation_notes.md — Example Package Generation and Manifests Implementation Notes](05_functionalities/04_example_package_generation_manifests/04_implementation_notes.md)

### 05_functionalities/05_generated_example_io_manual_composition/
- [01_quickstart.md — Generated Example I/O Manual Composition Quickstart](05_functionalities/05_generated_example_io_manual_composition/01_quickstart.md)
- [02_usage_guide.md — Generated Example I/O Manual Composition Usage Guide](05_functionalities/05_generated_example_io_manual_composition/02_usage_guide.md)
- [04_implementation_notes.md — Generated Example I/O Manual Composition Implementation Notes](05_functionalities/05_generated_example_io_manual_composition/04_implementation_notes.md)

### 05_functionalities/06_transcript_preprocessing_tabularization_chat_export/
- [01_quickstart.md — Transcript Preprocessing, Tabularization, Auto-Tabularization, and CHAT Export Quickstart](05_functionalities/06_transcript_preprocessing_tabularization_chat_export/01_quickstart.md)
- [02_usage_guide.md — Transcript Preprocessing, Tabularization, Auto-Tabularization, and CHAT Export Usage Guide](05_functionalities/06_transcript_preprocessing_tabularization_chat_export/02_usage_guide.md)
- [03_research_context.md — Transcript Preprocessing, Tabularization, Auto-Tabularization, and CHAT Export Research Context](05_functionalities/06_transcript_preprocessing_tabularization_chat_export/03_research_context.md)
- [04_implementation_notes.md — Transcript Preprocessing, Tabularization, Auto-Tabularization, and CHAT Export Implementation Notes](05_functionalities/06_transcript_preprocessing_tabularization_chat_export/04_implementation_notes.md)

### 05_functionalities/07_transcript_text_normalization_speaker_exclusion/
- [01_quickstart.md — Transcript Text Normalization and Speaker Exclusion Quickstart](05_functionalities/07_transcript_text_normalization_speaker_exclusion/01_quickstart.md)
- [02_usage_guide.md — Transcript Text Normalization and Speaker Exclusion Usage Guide](05_functionalities/07_transcript_text_normalization_speaker_exclusion/02_usage_guide.md)
- [03_research_context.md — Transcript Text Normalization and Speaker Exclusion Research Context](05_functionalities/07_transcript_text_normalization_speaker_exclusion/03_research_context.md)
- [04_implementation_notes.md — Transcript Text Normalization and Speaker Exclusion Implementation Notes](05_functionalities/07_transcript_text_normalization_speaker_exclusion/04_implementation_notes.md)

### 05_functionalities/08_metadata_extraction/
- [01_quickstart.md — Metadata Extraction Quickstart](05_functionalities/08_metadata_extraction/01_quickstart.md)
- [02_usage_guide.md — Metadata Extraction Usage Guide](05_functionalities/08_metadata_extraction/02_usage_guide.md)
- [03_research_context.md — Metadata Extraction Research Context](05_functionalities/08_metadata_extraction/03_research_context.md)
- [04_implementation_notes.md — Metadata Extraction Implementation Notes](05_functionalities/08_metadata_extraction/04_implementation_notes.md)

### 05_functionalities/09_configured_filenames_file_discovery_input_selection/
- [01_quickstart.md — Configured Filenames, File Discovery, and Input Selection Quickstart](05_functionalities/09_configured_filenames_file_discovery_input_selection/01_quickstart.md)
- [02_usage_guide.md — Configured Filenames, File Discovery, and Input Selection Usage Guide](05_functionalities/09_configured_filenames_file_discovery_input_selection/02_usage_guide.md)
- [04_implementation_notes.md — Configured Filenames, File Discovery, and Input Selection Implementation Notes](05_functionalities/09_configured_filenames_file_discovery_input_selection/04_implementation_notes.md)

### 05_functionalities/10_configurable_sample_utterance_identifiers/
- [01_quickstart.md — Configurable Sample and Utterance Identifiers Quickstart](05_functionalities/10_configurable_sample_utterance_identifiers/01_quickstart.md)
- [02_usage_guide.md — Configurable Sample and Utterance Identifiers Usage Guide](05_functionalities/10_configurable_sample_utterance_identifiers/02_usage_guide.md)
- [03_research_context.md — Configurable Sample and Utterance Identifiers Research Context](05_functionalities/10_configurable_sample_utterance_identifiers/03_research_context.md)
- [04_implementation_notes.md — Configurable Sample and Utterance Identifiers Implementation Notes](05_functionalities/10_configurable_sample_utterance_identifiers/04_implementation_notes.md)

### 05_functionalities/11_revision_handling/
- [01_quickstart.md — Revision Handling Quickstart](05_functionalities/11_revision_handling/01_quickstart.md)
- [02_usage_guide.md — Revision Handling Usage Guide](05_functionalities/11_revision_handling/02_usage_guide.md)
- [03_research_context.md — Revision Handling Research Context](05_functionalities/11_revision_handling/03_research_context.md)
- [04_implementation_notes.md — Revision Handling Implementation Notes](05_functionalities/11_revision_handling/04_implementation_notes.md)

### 05_functionalities/12_reliability_selection_evaluation_reselection/
- [01_quickstart.md — Reliability Selection, Evaluation, and Reselection Quickstart](05_functionalities/12_reliability_selection_evaluation_reselection/01_quickstart.md)
- [02_usage_guide.md — Reliability Selection, Evaluation, and Reselection Usage Guide](05_functionalities/12_reliability_selection_evaluation_reselection/02_usage_guide.md)
- [03_research_context.md — Reliability Selection, Evaluation, and Reselection Research Context](05_functionalities/12_reliability_selection_evaluation_reselection/03_research_context.md)
- [04_implementation_notes.md — Reliability Selection, Evaluation, and Reselection Implementation Notes](05_functionalities/12_reliability_selection_evaluation_reselection/04_implementation_notes.md)

### 05_functionalities/13_sample_subsetting_resubsetting/
- [01_quickstart.md — General Sample Subsetting and Re-Subsetting Quickstart](05_functionalities/13_sample_subsetting_resubsetting/01_quickstart.md)
- [02_usage_guide.md — General Sample Subsetting and Re-Subsetting Usage Guide](05_functionalities/13_sample_subsetting_resubsetting/02_usage_guide.md)
- [03_research_context.md — General Sample Subsetting and Re-Subsetting Research Context](05_functionalities/13_sample_subsetting_resubsetting/03_research_context.md)
- [04_implementation_notes.md — General Sample Subsetting and Re-Subsetting Implementation Notes](05_functionalities/13_sample_subsetting_resubsetting/04_implementation_notes.md)

### 05_functionalities/14_blinding_unblinding_auto_blind/
- [01_quickstart.md — Blinding, Unblinding, and Auto-Blind Quickstart](05_functionalities/14_blinding_unblinding_auto_blind/01_quickstart.md)
- [02_usage_guide.md — Blinding, Unblinding, and Auto-Blind Usage Guide](05_functionalities/14_blinding_unblinding_auto_blind/02_usage_guide.md)
- [03_research_context.md — Blinding, Unblinding, and Auto-Blind Research Context](05_functionalities/14_blinding_unblinding_auto_blind/03_research_context.md)
- [04_implementation_notes.md — Blinding, Unblinding, and Auto-Blind Implementation Notes](05_functionalities/14_blinding_unblinding_auto_blind/04_implementation_notes.md)

### 05_functionalities/15_speaking_time_rate_calculation/
- [01_quickstart.md — Speaking-Time Rate Calculation Quickstart](05_functionalities/15_speaking_time_rate_calculation/01_quickstart.md)
- [02_usage_guide.md — Speaking-Time Rate Calculation Usage Guide](05_functionalities/15_speaking_time_rate_calculation/02_usage_guide.md)
- [03_research_context.md — Speaking-Time Rate Calculation Research Context](05_functionalities/15_speaking_time_rate_calculation/03_research_context.md)
- [04_implementation_notes.md — Speaking-Time Rate Calculation Implementation Notes](05_functionalities/15_speaking_time_rate_calculation/04_implementation_notes.md)

### 05_functionalities/16_target_vocabulary_resource_management/
- [01_quickstart.md — Target Vocabulary Resource Management Quickstart](05_functionalities/16_target_vocabulary_resource_management/01_quickstart.md)
- [02_usage_guide.md — Target Vocabulary Resource Management Usage Guide](05_functionalities/16_target_vocabulary_resource_management/02_usage_guide.md)
- [03_research_context.md — Target Vocabulary Resource Management Research Context](05_functionalities/16_target_vocabulary_resource_management/03_research_context.md)
- [04_implementation_notes.md — Target Vocabulary Resource Management Implementation Notes](05_functionalities/16_target_vocabulary_resource_management/04_implementation_notes.md)

### 05_functionalities/17_powers_automation_support/
- [01_quickstart.md — POWERS Automation Support Quickstart](05_functionalities/17_powers_automation_support/01_quickstart.md)
- [02_usage_guide.md — POWERS Automation Support Usage Guide](05_functionalities/17_powers_automation_support/02_usage_guide.md)
- [03_research_context.md — POWERS Automation Support Research Context](05_functionalities/17_powers_automation_support/03_research_context.md)
- [04_implementation_notes.md — POWERS Automation Support Implementation Notes](05_functionalities/17_powers_automation_support/04_implementation_notes.md)

### 05_functionalities/18_coder_identifier_columns/
- [01_quickstart.md — Coder Identifier Columns Quickstart](05_functionalities/18_coder_identifier_columns/01_quickstart.md)
- [02_usage_guide.md — Coder Identifier Columns Usage Guide](05_functionalities/18_coder_identifier_columns/02_usage_guide.md)
- [03_research_context.md — Coder Identifier Columns Research Context](05_functionalities/18_coder_identifier_columns/03_research_context.md)
- [04_implementation_notes.md — Coder Identifier Columns Implementation Notes](05_functionalities/18_coder_identifier_columns/04_implementation_notes.md)

### 06_workflows/01_cli_project_setup_first_run/
- [01_quickstart.md — CLI Project Setup and First Run Quickstart](06_workflows/01_cli_project_setup_first_run/01_quickstart.md)
- [02_usage_guide.md — CLI Project Setup and First Run Usage Guide](06_workflows/01_cli_project_setup_first_run/02_usage_guide.md)
- [03_research_context.md — CLI Project Setup and First Run Research Context](06_workflows/01_cli_project_setup_first_run/03_research_context.md)
- [04_implementation_notes.md — CLI Project Setup and First Run Implementation Notes](06_workflows/01_cli_project_setup_first_run/04_implementation_notes.md)

### 06_workflows/02_web_app_project_setup_first_run/
- [01_quickstart.md — Web App Project Setup and First Run Quickstart](06_workflows/02_web_app_project_setup_first_run/01_quickstart.md)
- [02_usage_guide.md — Web App Project Setup and First Run Usage Guide](06_workflows/02_web_app_project_setup_first_run/02_usage_guide.md)
- [03_research_context.md — Web App Project Setup and First Run Research Context](06_workflows/02_web_app_project_setup_first_run/03_research_context.md)
- [04_implementation_notes.md — Web App Project Setup and First Run Implementation Notes](06_workflows/02_web_app_project_setup_first_run/04_implementation_notes.md)

### 06_workflows/03_example_dataset_command_specific_packages/
- [01_quickstart.md — Example Dataset and Command-Specific Packages Quickstart](06_workflows/03_example_dataset_command_specific_packages/01_quickstart.md)
- [02_usage_guide.md — Example Dataset and Command-Specific Packages Usage Guide](06_workflows/03_example_dataset_command_specific_packages/02_usage_guide.md)
- [03_research_context.md — Example Dataset and Command-Specific Packages Research Context](06_workflows/03_example_dataset_command_specific_packages/03_research_context.md)
- [04_implementation_notes.md — Example Dataset and Command-Specific Packages Implementation Notes](06_workflows/03_example_dataset_command_specific_packages/04_implementation_notes.md)

### 06_workflows/04_transcription_based_workflow_baseline/
- [01_quickstart.md — Transcription-Based Workflow Baseline Quickstart](06_workflows/04_transcription_based_workflow_baseline/01_quickstart.md)
- [02_usage_guide.md — Transcription-Based Workflow Baseline Usage Guide](06_workflows/04_transcription_based_workflow_baseline/02_usage_guide.md)
- [03_research_context.md — Transcription-Based Workflow Baseline Research Context](06_workflows/04_transcription_based_workflow_baseline/03_research_context.md)
- [04_implementation_notes.md — Transcription-Based Workflow Baseline Implementation Notes](06_workflows/04_transcription_based_workflow_baseline/04_implementation_notes.md)

### 06_workflows/05_transcription_reliability/
- [01_quickstart.md — Transcription Reliability Quickstart](06_workflows/05_transcription_reliability/01_quickstart.md)
- [02_usage_guide.md — Transcription Reliability Usage Guide](06_workflows/05_transcription_reliability/02_usage_guide.md)
- [03_research_context.md — Transcription Reliability Research Context](06_workflows/05_transcription_reliability/03_research_context.md)

### 06_workflows/06_transcript_table_revision_chat_export/
- [01_quickstart.md — Transcript Table Revision and CHAT Export Quickstart](06_workflows/06_transcript_table_revision_chat_export/01_quickstart.md)
- [02_usage_guide.md — Transcript Table Revision and CHAT Export Usage Guide](06_workflows/06_transcript_table_revision_chat_export/02_usage_guide.md)
- [03_research_context.md — Transcript Table Revision and CHAT Export Research Context](06_workflows/06_transcript_table_revision_chat_export/03_research_context.md)

### 06_workflows/07_monologic_narrative_complete_utterances/
- [01_quickstart.md — Monologic Narrative Complete Utterances Quickstart](06_workflows/07_monologic_narrative_complete_utterances/01_quickstart.md)
- [02_usage_guide.md — Monologic Narrative Complete Utterances Usage Guide](06_workflows/07_monologic_narrative_complete_utterances/02_usage_guide.md)
- [03_research_context.md — Monologic Narrative Complete Utterances Research Context](06_workflows/07_monologic_narrative_complete_utterances/03_research_context.md)

### 06_workflows/08_monologic_narrative_word_counting/
- [01_quickstart.md — Monologic Narrative Word Counting Quickstart](06_workflows/08_monologic_narrative_word_counting/01_quickstart.md)
- [02_usage_guide.md — Monologic Narrative Word Counting Usage Guide](06_workflows/08_monologic_narrative_word_counting/02_usage_guide.md)
- [03_research_context.md — Monologic Narrative Word Counting Research Context](06_workflows/08_monologic_narrative_word_counting/03_research_context.md)

### 06_workflows/09_monologic_narrative_target_vocabulary_coverage/
- [01_quickstart.md — Monologic Narrative Target Vocabulary Coverage Quickstart](06_workflows/09_monologic_narrative_target_vocabulary_coverage/01_quickstart.md)
- [02_usage_guide.md — Monologic Narrative Target Vocabulary Coverage Usage Guide](06_workflows/09_monologic_narrative_target_vocabulary_coverage/02_usage_guide.md)
- [03_research_context.md — Monologic Narrative Target Vocabulary Coverage Research Context](06_workflows/09_monologic_narrative_target_vocabulary_coverage/03_research_context.md)

### 06_workflows/10_digital_conversational_turns/
- [01_quickstart.md — Digital Conversational Turns Quickstart](06_workflows/10_digital_conversational_turns/01_quickstart.md)
- [02_usage_guide.md — Digital Conversational Turns Usage Guide](06_workflows/10_digital_conversational_turns/02_usage_guide.md)
- [03_research_context.md — Digital Conversational Turns Research Context](06_workflows/10_digital_conversational_turns/03_research_context.md)

### 06_workflows/11_monologic_narrative_integrated_workflow/
- [01_quickstart.md — Monologic Narrative Integrated Workflow Quickstart](06_workflows/11_monologic_narrative_integrated_workflow/01_quickstart.md)
- [02_usage_guide.md — Monologic Narrative Integrated Workflow Usage Guide](06_workflows/11_monologic_narrative_integrated_workflow/02_usage_guide.md)
- [03_research_context.md — Monologic Narrative Integrated Workflow Research Context](06_workflows/11_monologic_narrative_integrated_workflow/03_research_context.md)

### 06_workflows/12_clinician_client_dialogue_powers/
- [01_quickstart.md — Clinician-Client Dialogue POWERS Quickstart](06_workflows/12_clinician_client_dialogue_powers/01_quickstart.md)
- [02_usage_guide.md — Clinician-Client Dialogue POWERS Usage Guide](06_workflows/12_clinician_client_dialogue_powers/02_usage_guide.md)
- [03_research_context.md — Clinician-Client Dialogue POWERS Research Context](06_workflows/12_clinician_client_dialogue_powers/03_research_context.md)

### Manual root
- [99_references.md — References](99_references.md)

---

## Notes

- Regenerate this file after adding or renaming manual sections.
- Keep numeric prefixes stable to preserve predictable ordering.
- This outline is a derived support artifact for navigation and build workflows.
