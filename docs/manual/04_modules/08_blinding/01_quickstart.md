# Blinding Module Quickstart

The Blinding module encodes and decodes configured identifier columns in tabular files. It supports standalone blinding commands and also informs auto-blind behavior in some coding workflows.

## Commands

| Command | Main use |
|---|---|
| `diaad blinding encode` | Replace configured identifiers with blind codes and write diagnostics. |
| `diaad blinding decode` | Restore identifiers from a blind codebook. |

## Typical Use

Use blinding when coders should work with masked identifiers:

```bash
diaad blinding encode --config config
```

After manual coding, decode when analysis should be tied back to canonical sample identifiers:

```bash
diaad blinding decode --config config
```

Project settings live in `advanced.yaml`, especially:

```yaml
auto_blind: false
blind_columns:
  - sample_id
```

## Common Outputs

| Command | Typical outputs |
|---|---|
| `blinding encode` | blinded workbook, blinding diagnostics, `blind_codebook.xlsx` |
| `blinding decode` | decoded workbook |

## Read Next

- [Configuration](../../02_operation/04_configuration.md)
- [Exact file name matching](../../03_features/03_exact_file_name_matching.md)
- [Web app operation](../../02_operation/03_webapp.md)

Later command and functionality pages describe codebook discovery, auto-blind behavior, and privacy/de-identification cautions in more detail.
