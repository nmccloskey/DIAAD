# Blinding Module Quickstart

The Blinding module encodes and decodes configured identifier columns in tabular files. It supports standalone blinding commands and also informs auto-blind behavior in some coding workflows.

## Commands

| Command | Main use |
|---|---|
| `diaad blinding encode` | Replace configured identifiers with blind codes and write diagnostics. |
| `diaad blinding decode` | Restore identifiers from a blind codebook. |

## Typical Use

Use blinding when coders or downstream analysts should work with masked identifiers. For manual coding, encode before distributing coder-facing files, or use a coding workflow's `auto_blind` support when available:

```bash
diaad blinding encode --config config
```

After manual coding, decode before DIAAD analysis when analysis needs canonical sample identifiers, metadata joins, or exact filename/material matching:

```bash
diaad blinding decode --config config
```

After analysis, a project may choose to encode selected analysis exports again for blinded statistical workflows or external sharing. That post-analysis encoding should use the same codebook when the blinded values need to remain comparable with earlier blinded materials.

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

- Configuration: `docs/manual/02_operation/04_configuration.md`
- Exact file name matching: `docs/manual/03_features/03_exact_file_name_matching.md`
- Web app operation: `docs/manual/02_operation/03_webapp.md`

Command and functionality pages describe codebook discovery, auto-blind behavior, and privacy/de-identification cautions in more detail.
