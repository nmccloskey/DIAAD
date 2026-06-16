# Generated Example I/O Manual Composition Quickstart

Generated Example I/O pages are DIAAD's generated manual view for command examples. They live outside the authored manual tree, but they carry front matter that lets a composition layer place them beside authored command views.

## Source Roots

Authored manual pages live under:

```text
docs/manual/
```

Generated Example I/O pages live under:

```text
src/diaad/examples/assets/rendered_docs/example_io/
```

Do not copy generated Example I/O pages into the authored manual tree. They should be composed by metadata.

## User-Facing Meaning

An authored command page may provide:

```text
Quickstart
Usage Guide
Implementation Notes
```

The generated Example I/O page provides:

```text
Example I/O
```

Together, these views help users understand both how to run a command and what its files look like.

## Regeneration

Documentation maintainers can regenerate Example I/O markdown with:

```bash
diaad examples --render-docs
```

Ordinary users do not need this command. They usually want `diaad examples` or `diaad examples --for-command "<command>"` instead.

## Read Next

- Generated Example I/O feature: `docs/manual/03_features/04_generated_example_io.md`
- Example package generation and manifests: `docs/manual/05_functionalities/04_example_package_generation_manifests/01_quickstart.md`
- Examples module: `docs/manual/04_modules/09_examples/01_quickstart.md`
- Examples command implementation notes: `docs/manual/04_modules/09_examples/05_commands/01_examples/04_implementation_notes.md`
