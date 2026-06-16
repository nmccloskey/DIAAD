# Testing

DIAAD uses `pytest` for unit and regression testing. The tests are primarily developer-facing: they check command parsing, configuration handling, file discovery, transcript processing, coding utilities, example generation, web-app command exposure, and module-specific data transformations. They do not replace methodological validation of a study protocol or human review of discourse-analysis outputs.

## Installing Test Dependencies

The development dependency group includes `pytest`:

```bash
pip install -e ".[dev]"
```

When testing web or NLP-backed behavior, install the relevant extras in the same environment:

```bash
pip install -e ".[dev,web,nlp]"
python -m spacy download en_core_web_sm
```

The repository is currently configured for Python `>=3.12,<3.13`, so tests should be run in a Python 3.12 environment.

## Test Folder Structure

Tests live under `tests/` and are organized by package area or user-facing subsystem.

```text
tests/
  conftest.py
  helpers.py
  test_main.py
  test_webapp_command_menu.py
  test_webapp_launcher.py
  test_blinding/
  test_cli/
  test_coding/
    test_compl_utts/
    test_convo_turns/
    test_powers/
    test_target_vocab/
    test_templates/
    test_utils/
    test_word_counts/
  test_core/
  test_examples/
  test_metadata/
  test_transcripts/
```

Current coverage is concentrated in these areas:

| Test area | Typical focus |
|---|---|
| `test_cli/` | Command registry, parser behavior, dispatch prerequisites. |
| `test_core/` | Configuration, CLI overrides, run context, provenance artifacts. |
| `test_metadata/` | File discovery, metadata utilities, blinding and unblinding helpers. |
| `test_transcripts/` | CHAT loading, transcript table construction, detabularization. |
| `test_coding/test_templates/` | Generic templates, sample subsetting, identifier handling. |
| `test_coding/test_compl_utts/` | Complete Utterance identifier behavior. |
| `test_coding/test_word_counts/` | Word-count file generation and configurable identifiers. |
| `test_coding/test_powers/` | POWERS automation and identifier behavior. |
| `test_coding/test_target_vocab/` | Target vocabulary resources, utilities, files, analysis, identifiers. |
| `test_coding/test_convo_turns/` | Digital Conversational Turns analysis and identifiers. |
| `test_coding/test_utils/` | Shared sampling, rates, coders, reselection, transcript helpers. |
| `test_blinding/` | Standalone blinding command behavior. |
| `test_examples/` | Example package generation and rendered/generated documentation behavior. |
| `test_webapp_command_menu.py` | Web app command menu exposure and examples exclusion from ordinary command selection. |

`__pycache__/` directories are generated Python cache directories and should not be treated as test source files.

## Naming Conventions

Pytest discovers tests by standard naming conventions:

- Test files use `test_*.py`.
- Test functions use `test_*`.
- Shared fixtures belong in `tests/conftest.py` when they are broadly useful.
- Small test helpers belong in `tests/helpers.py` or a focused helper module.
- Subdirectories use `test_<area>/` names when they group related module tests.

When adding a feature, command, or behavior, add tests near the source area most directly affected. For example, parser changes belong under `tests/test_cli/`, config changes under `tests/test_core/`, and target vocabulary behavior under `tests/test_coding/test_target_vocab/`.

## Running Tests

From a normal activated development environment, the basic command is:

```bash
python -m pytest tests
```

On Windows in this repository, the preferred helper is:

```powershell
.\scripts\run_tests.ps1 tests
```

For focused tests, pass the test file or directory:

```powershell
.\scripts\run_tests.ps1 tests/test_transcripts/test_detabularization.py
.\scripts\run_tests.ps1 tests/test_coding/test_target_vocab
```

Additional pytest arguments can be passed through the helper:

```powershell
.\scripts\run_tests.ps1 tests/test_cli -q
```

## Windows Test Helper

The tracked helper script is `scripts/run_tests.ps1`.

Its behavior is:

1. Accept pytest arguments, defaulting to `tests`.
2. Use `$env:DIAAD_PYTHON` if it is set.
3. Otherwise use the expected Conda environment Python at `%USERPROFILE%\anaconda3\envs\diaad\python.exe`.
4. Run `python -m pytest` with the provided arguments.
5. Exit with pytest's exit code.

This makes test execution more predictable in Codex and Windows PowerShell sessions than calling Conda wrappers directly.

To use a different Python executable:

```powershell
$env:DIAAD_PYTHON = "C:\path\to\python.exe"
.\scripts\run_tests.ps1 tests
```

## Codex Testing Protocol

The local command notes in `.codex-local/COMMANDS.md` define the preferred testing pattern for Codex sessions:

```powershell
.\scripts\run_tests.ps1 tests
```

For focused tests:

```powershell
.\scripts\run_tests.ps1 tests/test_transcripts/test_detabularization.py
```

Avoid `conda run -n diaad pytest ...` in Codex sessions unless Conda wrapper behavior itself is being tested. If sandboxed Python spawning fails before pytest starts, retry once. If the same failure persists and the test is important for the requested work, request escalated execution of the same helper command. Do not report test success unless pytest actually ran and returned success.

## General Testing Policy

Tests should be small, deterministic, and source-grounded. Prefer synthetic fixtures, temporary directories, and explicit assertions over large end-to-end test setups.

When changing DIAAD behavior, update or add tests for:

- command registration, parser syntax, and dispatch behavior when user-facing commands change;
- configuration defaults, override behavior, and validation when config fields change;
- exact file discovery and output naming when filenames or input requirements change;
- generated example specs, manifests, and rendered docs when examples change;
- workbook sheets, required columns, and identifier behavior when coding workflows change;
- blinding, unblinding, and provenance behavior when run artifacts or privacy-related behavior changes.

Tests should not use identifiable study data. Synthetic examples are preferred, and temporary filesystem paths should be used for generated artifacts.

## What Tests Can and Cannot Prove

Passing tests indicate that the current implementation satisfies the checked software expectations. They do not prove that:

- a coding protocol is methodologically valid;
- a reliability threshold is appropriate for a study;
- automated first-pass coding is sufficient without human review;
- a workflow is de-identified or safe to upload to a hosted app.

Those decisions require project-specific review, methodological judgment, and applicable privacy or compliance procedures.
