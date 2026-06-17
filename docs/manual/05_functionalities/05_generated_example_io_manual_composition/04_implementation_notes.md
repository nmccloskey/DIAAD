# Generated Example I/O Manual Composition Implementation Notes

Generated Example I/O composition is implemented by the example renderer and by metadata conventions in the generated Markdown files.

## Source Anchors

Primary sources:

- `src/diaad/examples/render_docs.py`
- `src/diaad/examples/generate.py`
- `src/diaad/examples/__init__.py`
- `src/diaad/webapp/streamlit_app.py`
- `src/diaad/examples/assets/rendered_docs/example_io/`
- `.codex-local/itineraries/DIAAD_oriented_PSAIR_notes_for_user_example_manual_synthesis.md`

Relevant tests:

- `tests/test_examples/test_examples.py`
- `tests/test_webapp_command_menu.py`

## Rendered Docs Root

`src/diaad/examples/__init__.py` exposes helper functions for packaged generated docs:

```text
get_example_io_docs_path()
iter_example_io_markdown_files()
```

These helpers use:

```text
DOC_PACKAGE = diaad.examples
DOC_ROOT = assets/rendered_docs/example_io
```

The web app passes the authored manual root and generated Example I/O root to PSAIR's composed manual viewer, so users browse one manual tree instead of separate manual and example menus.

## Renderer Metadata

`render_docs.py` defines the generated view identity:

```text
view: example_io
view_label: Example I/O
view_order: 50
slot: examples
source_manual: generated_example_io
generated: true
```

Command pages are wrapped with command front matter by `_with_command_front_matter()`. The overview is wrapped by `_with_overview_front_matter()`.

Command page front matter includes:

- `object_type: command`;
- `object_types: [command]`;
- `object_id`;
- `command_id`;
- `canonical_command`;
- `module_id`;
- `title`;
- generated view metadata.

The full overview page includes:

- `object_type: workflow`;
- `object_types: [workflow, command]`;
- `object_id: full_example_dataset`;
- `workflow_id: full_example_dataset`;
- `command_id: examples`;
- `command_subtype: omnibus`;
- `typology: omnibus_command_workflow`;
- generated view metadata.

## Rendering Flow

`render_example_docs()`:

1. reads packaged example specs;
2. creates a scratch synthetic project;
3. calls `generate_example_files()` for the full example project;
4. builds command page bodies using `COMMAND_DOC_BUILDERS`;
5. writes the generated overview and command pages under the rendered docs root.

`_validate_command_doc_builders()` ensures the command doc builders match the example command plan registry. This helps prevent a command from having a downloadable example package but no generated Example I/O page, or the reverse.

## Runtime Path Rendering

Generated command docs are body-authored against full-dataset expected-output paths in some renderer helpers, then `_with_command_front_matter()` calls `render_runtime_preview_paths()` to replace expected-output atlas links with normal runtime display paths.

This preserves the full-dataset atlas for the package README and manifest while presenting command pages in user-facing runtime terms.

## Composition Contract

The intended composition key is:

```text
object_type + object_id
```

For command pages, `object_id` intentionally duplicates `command_id` so a generic composer does not need DIAAD-specific command logic.

Recommended shared view vocabulary:

```text
quickstart
usage_guide
research_context
implementation_notes
example_io
```

Authored DIAAD pages live under `docs/manual/`; generated Example I/O pages live under the packaged example docs root. PSAIR's composer scans both roots and groups by front matter rather than moving files. During the transition, DIAAD-compatible path inference can help authored command folders that do not yet carry front matter, but explicit metadata is the clearer long-term contract.

## Maintenance Checks

After renderer or generator changes, verify:

- every `EXAMPLE_COMMAND_PLANS` command has one generated command page;
- every generated command page has `view: example_io`;
- command pages use runtime-shaped output paths;
- only the full overview or full-package README needs `expected_outputs/` atlas language;
- generated front matter remains parseable YAML;
- `diaad examples --render-docs` updates the rendered docs cleanly.

Use the test helper from Testing (`docs/manual/02_operation/05_testing.md`) when code or generated artifacts change:

```powershell
.\scripts\run_tests.ps1 tests/test_examples/test_examples.py
```

## Boundary With PSAIR

DIAAD should keep generating rich Example I/O pages with stable metadata. PSAIR owns generalized multi-root manual composition for viewing and export. This keeps DIAAD from copying generated files into `docs/manual/` and keeps PSAIR from needing to scrape DIAAD-specific body text. Detailed composer behavior, virtual paths, unmatched generated pages, and media-link limitations belong in PSAIR's doctools manual rather than in DIAAD's user manual.

## Read Next

- Example package generation implementation notes: `docs/manual/05_functionalities/04_example_package_generation_manifests/04_implementation_notes.md`
- Generated Example I/O feature: `docs/manual/03_features/04_generated_example_io.md`
- Testing: `docs/manual/02_operation/05_testing.md`
