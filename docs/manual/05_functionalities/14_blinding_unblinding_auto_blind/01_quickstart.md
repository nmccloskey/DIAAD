# Blinding, Unblinding, and Auto-Blind Quickstart

DIAAD blinding replaces configured identifier values with blind codes and preserves a codebook so those values can be restored later.

## Typical Workflow

For manual coding workflows, the recommended sequence is:

```text
prepare coding files
encode or auto-blind identifiers
manual coding on blinded files
decode back to canonical identifiers
DIAAD analysis
optional post-analysis encoding for blinded exports
```

Encoding before manual coding can reduce coder access to sample identifiers. Decoding before DIAAD analysis helps analysis commands work with canonical sample identifiers and metadata relationships.

## Key Commands

```bash
diaad blinding encode
diaad blinding decode
```

Some coding-file generation commands can also use configured auto-blind behavior.

## Key Settings

Blinding settings live in `advanced.yaml`:

```yaml
advanced:
  auto_blind: false
  blind_columns:
    - sample_id
  id_columns:
    - sample_id
    - utterance_id
  codebook_filename: ''
```

Keep the blind codebook under controlled storage. It is needed for decoding and can defeat masking if shared too broadly.

## Read Next

- Blinding module: `docs/manual/04_modules/08_blinding/`
- `blinding encode` command: `docs/manual/04_modules/08_blinding/05_commands/01_encode/01_quickstart.md`
- `blinding decode` command: `docs/manual/04_modules/08_blinding/05_commands/02_decode/01_quickstart.md`
- Configuration: `docs/manual/02_operation/04_configuration.md`
