# Instruction Manual

**Version:** 0.0.0
**Generated:** 2026-06-14

---

## Manual Map (Tree)

```
├── 01_overview/
│   ├── 01_introduction.md — Introduction to DIAAD
│   ├── 02_methodolgical_overview.md — Methodological Overview
│   └── 03_functional_overview.md — Functional Overview
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
│   │       └── 04_subset/
│   │           ├── 01_quickstart.md — `templates subset` Quickstart
│   │           ├── 02_usage_guide.md — `templates subset` Usage Guide
│   │           └── 04_implementation_notes.md — `templates subset` Implementation Notes
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
│   │   └── 04_implementation_notes.md — POWERS Implementation Notes
│   ├── 06_target_vocabulary_coverage/
│   │   ├── 01_quickstart.md — Target Vocabulary Coverage Module Quickstart
│   │   ├── 03_research_context.md — Target Vocabulary Coverage Research Context
│   │   └── 04_implementation_notes.md — Target Vocabulary Coverage Implementation Notes
│   ├── 07_digital_conversational_turns/
│   │   ├── 01_quickstart.md — Digital Conversational Turns Module Quickstart
│   │   ├── 03_research_context.md — Digital Conversational Turns Research Context
│   │   └── 04_implementation_notes.md — Digital Conversational Turns Implementation Notes
│   ├── 08_blinding/
│   │   ├── 01_quickstart.md — Blinding Module Quickstart
│   │   ├── 03_research_context.md — Blinding Research Context
│   │   └── 04_implementation_notes.md — Blinding Implementation Notes
│   └── 09_examples/
│       ├── 01_quickstart.md — Examples Module Quickstart
│       ├── 02_usage_guide.md — Examples Module Usage Guide
│       ├── 03_research_context.md — Examples Research Context
│       └── 04_implementation_notes.md — Examples Implementation Notes
└── 99_references.md — References
```

## Outline (Links)

### 01_overview/
- [01_introduction.md — Introduction to DIAAD](01_overview/01_introduction.md)
- [02_methodolgical_overview.md — Methodological Overview](01_overview/02_methodolgical_overview.md)
- [03_functional_overview.md — Functional Overview](01_overview/03_functional_overview.md)

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

### 04_modules/06_target_vocabulary_coverage/
- [01_quickstart.md — Target Vocabulary Coverage Module Quickstart](04_modules/06_target_vocabulary_coverage/01_quickstart.md)
- [03_research_context.md — Target Vocabulary Coverage Research Context](04_modules/06_target_vocabulary_coverage/03_research_context.md)
- [04_implementation_notes.md — Target Vocabulary Coverage Implementation Notes](04_modules/06_target_vocabulary_coverage/04_implementation_notes.md)

### 04_modules/07_digital_conversational_turns/
- [01_quickstart.md — Digital Conversational Turns Module Quickstart](04_modules/07_digital_conversational_turns/01_quickstart.md)
- [03_research_context.md — Digital Conversational Turns Research Context](04_modules/07_digital_conversational_turns/03_research_context.md)
- [04_implementation_notes.md — Digital Conversational Turns Implementation Notes](04_modules/07_digital_conversational_turns/04_implementation_notes.md)

### 04_modules/08_blinding/
- [01_quickstart.md — Blinding Module Quickstart](04_modules/08_blinding/01_quickstart.md)
- [03_research_context.md — Blinding Research Context](04_modules/08_blinding/03_research_context.md)
- [04_implementation_notes.md — Blinding Implementation Notes](04_modules/08_blinding/04_implementation_notes.md)

### 04_modules/09_examples/
- [01_quickstart.md — Examples Module Quickstart](04_modules/09_examples/01_quickstart.md)
- [02_usage_guide.md — Examples Module Usage Guide](04_modules/09_examples/02_usage_guide.md)
- [03_research_context.md — Examples Research Context](04_modules/09_examples/03_research_context.md)
- [04_implementation_notes.md — Examples Implementation Notes](04_modules/09_examples/04_implementation_notes.md)

### Manual root
- [99_references.md — References](99_references.md)

---

## Notes

- Regenerate this file after adding or renaming manual sections.
- Keep numeric prefixes stable to preserve predictable ordering.
- This outline is a derived support artifact for navigation and build workflows.
