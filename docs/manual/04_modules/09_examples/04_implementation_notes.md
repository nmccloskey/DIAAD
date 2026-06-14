# Examples Implementation Notes

The Examples module is implemented under `src/diaad/examples/`.

## Generator

`generate.py` defines command IDs, package slugs, expected output paths, manifest structure, and the logic for full-dataset and command-specific example packages.

Full example packages and command-specific packages are generated from packaged specs and deterministic synthetic assets. Command-specific packages include a manifest that records command IDs, required capabilities, artifact paths, and suggested invocation.

## Rendered Example I/O

`render_docs.py` builds the generated `example_io` manual view. Command pages receive front matter with fields such as `object_type`, `object_id`, `command_id`, `canonical_command`, `module_id`, `view`, `view_order`, `slot`, and `source_manual`.

The full example dataset overview is modeled primarily as a workflow object with secondary command identity for `examples`.

## Web App Integration

The web app displays packaged Example I/O through the manual viewer and provides a dedicated examples download action. The selectable command menu is built from normal dispatch commands; `examples` is handled separately.

## Tests

Example tests verify package generation, manifest fields, command-plan coverage, rendered documentation front matter, and runtime-shaped output paths.

## Boundaries

Generated docs live outside `docs/manual/`. Authored manual pages should link to or describe the generated view, not duplicate generated previews.
