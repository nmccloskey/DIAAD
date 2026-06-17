# Generated Example I/O Manual Composition Usage Guide

Generated Example I/O pages and downloadable example packages support the same goal, but they are different artifacts.

## What Generated Example I/O Is

Generated Example I/O pages are Markdown documentation pages. They show:

- a command;
- a minimal project tree;
- basic configuration snippets;
- representative input snippets;
- output paths;
- preview tables or text excerpts;
- notes about synthetic example behavior.

They are designed to be read inside the manual or web app. They are generated from packaged specs and generated example files, so their content is reproducible.

## What Downloadable Packages Are

Downloadable example packages are file sets. They contain runnable or inspectable synthetic project material:

- config files;
- synthetic inputs;
- representative outputs;
- package README files;
- manifest files;
- illustrative command-specific logs.

They are designed for hands-on exploration.

The generated docs and downloadable packages do not need to match byte-for-byte. The docs are optimized for reading; the packages are optimized for trying the file structure locally.

## Composition By Metadata

Generated Example I/O pages include YAML front matter. Command pages declare identity such as:

```yaml
object_type: command
object_id: cus.files
command_id: cus.files
canonical_command: cus files
module_id: cus
view: example_io
view_label: Example I/O
view_order: 50
slot: examples
source_manual: generated_example_io
generated: true
```

The full example dataset overview is modeled differently because `diaad examples` is both an invokable command and an omnibus workflow/collection:

```yaml
object_type: workflow
object_types:
  - workflow
  - command
object_id: full_example_dataset
workflow_id: full_example_dataset
command_id: examples
canonical_command: examples
view: example_io
```

PSAIR's composition layer groups authored and generated files by object identity, then orders views by `view_order`. DIAAD keeps only the DIAAD metadata expectations here; the detailed composer API and export contract live in PSAIR's doctools manual.

## Expected Composed Shape

The intended user-facing shape is:

```text
Transcripts
  Tabularize
    Quickstart
    Usage Guide
    Implementation Notes
    Example I/O
```

The generated Example I/O file stays in the generated source root. The composed manual can still display it next to authored views, and those authored views do not have to be a complete fixed set for every command.

## Path Behavior

Generated command Example I/O pages should display normal runtime-shaped paths, such as:

```text
diaad_data/output/diaad_YYMMDD_HHMM/cu_coding/cu_coding.xlsx
```

The full example dataset may still describe `expected_outputs/` because that package is an atlas. Command pages should not present `expected_outputs/` as if it were a normal run output root.

## Manual Maintenance

When generated Example I/O content needs to change:

1. update the relevant example spec, generator, renderer, or test;
2. regenerate Example I/O;
3. inspect the rendered Markdown;
4. let the composed manual consume the generated view.

Do not paste generated preview tables or workbook snippets into authored pages.

## Read Next

- Generated Example I/O feature: `docs/manual/03_features/04_generated_example_io.md`
- Example package generation usage guide: `docs/manual/05_functionalities/04_example_package_generation_manifests/02_usage_guide.md`
- Testing: `docs/manual/02_operation/05_testing.md`
