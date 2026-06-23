# Target Vocabulary Resource Management Implementation Notes

Target-vocabulary resources are loaded, checked, and used by the `diaad.coding.target_vocab` package.

## Source Anchors

Primary sources:

- `src/diaad/coding/target_vocab/resources.py`
- `src/diaad/coding/target_vocab/files.py`
- `src/diaad/coding/target_vocab/utils.py`
- `src/diaad/coding/target_vocab/analysis.py`
- `src/diaad/coding/target_vocab/rates.py`
- `src/diaad/coding/target_vocab/resources/*.json`
- `src/diaad/core/config.py`
- `src/diaad/core/run_context.py`

Relevant tests:

- `tests/test_coding/test_target_vocab/test_resources.py`
- `tests/test_coding/test_target_vocab/test_files.py`
- `tests/test_coding/test_target_vocab/test_analysis.py`
- `tests/test_coding/test_target_vocab/test_utils.py`
- `tests/test_coding/test_target_vocab/test_identifiers.py`

## Resource Contract

`resources.py` requires:

```text
resource_id
display_name
language
task_type
base_forms
variant_map
```

The validator enforces non-empty string metadata, a non-empty duplicate-free `base_forms` list, a dictionary-shaped `variant_map`, and one-to-one variant assignment. Variants must be non-empty strings, must be listed under an existing base form, must not also be another base form, and must not map to more than one base form.

The optional `norms` block is validated as a mapping of metric names to CSV specifications. Current norm specifications require:

```text
url
format
columns
```

Only CSV norm specifications are supported. Norm column mappings must include `raw_score`, `group`, `pwa_percentile`, and `control_percentile`.

## Loading Behavior

`load_target_vocabulary_resources()` loads built-ins from `src/diaad/coding/target_vocab/resources/`. If `advanced.target_vocabulary_resource_path` is blank, the built-in set is the active set.

If a custom path is configured, DIAAD loads built-ins first and then custom resources. Custom resource IDs override built-in resource IDs. A custom path can be one JSON file or a directory of JSON files. Duplicate resource IDs inside the custom path raise an error.

## Command Outputs

`vocab file` writes:

```text
target_vocab/target_vocabulary_resource_template.json
```

`vocab check` writes:

```text
target_vocab/target_vocab_resource_check.txt
```

`vocab analyze` writes a timestamped workbook:

```text
target_vocab/target_vocab_data_YYMMDD_HHMM.xlsx
```

The analysis workbook contains `summary` and `details` sheets.

## Analysis Integration

`vocab analyze` prefers unblinded utterance data when available:

```text
*unblind_utterance_data*.xlsx
```

If that file is unavailable, it falls back to the configured transcript table filename. The input must provide a sample identifier, a stimulus or narrative column matching active resource IDs, utterance text, and speaking time when transcript-table fallback is used.

During analysis, DIAAD normalizes text, expands supported contractions, strips common CHAT/CLAN annotations, maps tokens through the reverse variant lookup, and aggregates sample-level summary metrics plus base-form detail rows.

## Boundary

The implementation validates resource structure and computes coverage. It does not validate the scientific design of a resource, audit the appropriateness of declared norms, or resolve ambiguous linguistic decisions that should be handled in a project protocol.

## Read Next

- `vocab analyze` implementation notes: `docs/manual/04_modules/06_target_vocabulary_coverage/05_commands/03_analyze/04_implementation_notes.md`
- Configuration implementation notes: `docs/manual/02_operation/04_configuration.md`
- Testing: `docs/manual/02_operation/05_testing.md`
