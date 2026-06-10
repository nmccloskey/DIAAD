# Windows Test Execution

On Windows, prefer invoking pytest through the DIAAD environment's Python executable rather than through `conda run`. This avoids extra wrapper process setup, which can be unreliable in sandboxed command runners.

Use the project helper script for normal test runs:

```powershell
.\scripts\run_tests.ps1
```

Pass pytest targets or flags directly to the script for focused runs:

```powershell
.\scripts\run_tests.ps1 tests/test_transcripts/test_detabularization.py
.\scripts\run_tests.ps1 tests/test_transcripts/test_detabularization.py -q
```

The script defaults to:

```powershell
$env:USERPROFILE\anaconda3\envs\diaad\python.exe
```

If your DIAAD environment lives somewhere else, set `DIAAD_PYTHON` before running tests:

```powershell
$env:DIAAD_PYTHON = "C:\path\to\envs\diaad\python.exe"
.\scripts\run_tests.ps1 tests
```

## Codex And Sandbox Notes

When running tests in Codex on Windows:

- Do not use `conda run -n diaad pytest ...` unless there is a specific reason to test Conda wrapper behavior.
- Prefer `.\scripts\run_tests.ps1 <pytest args>` or direct invocation with `python.exe -m pytest`.
- If Python process spawning fails before pytest starts, retry once.
- If the same spawn failure repeats but simple PowerShell reads still work, treat it as a sandbox/process-runner issue and request an escalated/non-sandboxed test run.
- Only report tests as passing after pytest has actually completed and returned success.
