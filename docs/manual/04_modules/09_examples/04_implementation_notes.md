# Examples Implementation Notes

The Examples module is implemented under `src/diaad/examples/`.

## Generator

`generate.py` defines command IDs, package slugs, expected output paths, manifest structure, and the logic for full-dataset and command-specific example packages.

Full example packages and command-specific packages are generated from packaged specs and deterministic synthetic assets. The full package is generated at `example_files_full_dataset/` and uses `config/`, `input/`, and `expected_outputs/`. Command-specific packages use `example_files_<slug>/` with `example_config/`, `example_input/`, `example_output/`, and `example_logs/`.

Command-specific packages include a manifest that records command IDs, required capabilities, artifact paths, runtime display paths, and suggested invocation.

## Rendered Example I/O

`render_docs.py` builds the generated `example_io` manual view. Command pages receive front matter with fields such as `object_type`, `object_id`, `command_id`, `canonical_command`, `module_id`, `view`, `view_order`, `slot`, and `source_manual`.

The full example dataset overview is modeled primarily as a workflow object with secondary command identity for `examples`.

## Web App Integration

The web app displays packaged Example I/O through the manual viewer and provides a dedicated examples download action in Part 3: Commands. The selectable command menu is built from normal dispatch commands; `examples` is handled separately.

When no commands are selected, the web app generates a ZIP containing `example_files_full_dataset/`. When commands are selected, it generates a ZIP containing the matching command-specific `example_files_<slug>/` package.

## Tests

Example tests verify package generation, manifest fields, command-plan coverage, rendered documentation front matter, and runtime-shaped output paths.

## Boundaries

Generated docs live outside `docs/manual/`. Authored manual pages should link to or describe the generated view, not duplicate generated previews.
