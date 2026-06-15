# `examples` Quickstart

`diaad examples` generates synthetic DIAAD example files for learning, testing an installation, and inspecting expected project structure.

## Run

Generate the full example dataset:

```bash
diaad examples --config config
```

Generate an example package for one command:

```bash
diaad examples --for-command "vocab check" --config config
```

Generate one package for multiple specific commands by repeating `--for-command`:

```bash
diaad examples --for-command "cus analyze" --for-command "cus evaluate" --config config
```

## Primary Outputs

The full example dataset is written inside the current timestamped DIAAD output directory as:

```text
example_files_full_dataset/
```

Command-specific examples are written as:

```text
example_files_<slug>/
```

For example:

```text
example_files_cus_analyze_cus_evaluate/
```

## Web App

In the web app, use Part 3: Commands. If no commands are selected, the examples button downloads the full example dataset. If commands are selected, it downloads command-specific examples for those selected functions.

## Immediate Next Step

Open the package README and `example_manifest.json`, then inspect the generated configuration and input folders before comparing the representative outputs.

## Read Next

- `examples` usage guide: `docs/manual/04_modules/09_examples/05_commands/01_examples/02_usage_guide.md`
- Examples module usage guide: `docs/manual/04_modules/09_examples/02_usage_guide.md`
- Generated Example I/O: `docs/manual/03_features/04_generated_example_io.md`
- Configuration: `docs/manual/02_operation/04_configuration.md`
