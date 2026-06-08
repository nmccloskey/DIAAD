# CU Coding Analysis Example

This example demonstrates how `diaad cus analyze` summarizes filled complete-utterance coding by utterance and by sample.

## Command

```bash
diaad cus analyze --config config
```

## Project Files

```
your_project/
  config/
    project.yaml
    advanced.yaml
  diaad_data/
    input/
      cu_coding/
        cu_coding.xlsx
        cu_blind_codebook.xlsx
    output/
      diaad_YYMMDD_HHMM/
        cu_coding_analysis/
          cu_coding_by_utterance.xlsx
          cu_coding_by_sample_long.xlsx
          cu_coding_by_sample.xlsx
        logs/
          diaad_YYMMDD_HHMM.log
          diaad_YYMMDD_HHMM_metadata.json
```

## Basic Config

```yaml
input_dir: diaad_data/input
output_dir: diaad_data/output
```

## Advanced Config

```yaml
auto_blind: true
blind_columns:
- sample_id
metadata_source: transcript_tables
codebook_filename: ''
```

## Input Snippet

The command reads `diaad_data/input/cu_coding/cu_coding.xlsx`. The blind codebook is included so analysis outputs can recover sample identifiers.

## Output Preview

`expected_outputs/cus_module/cus_analyze/cu_coding_by_utterance.xlsx`

| input_order | shuffled_order | stimulus | utterance_id | position | position_sub | speaker | utterance | sv | rel | cu | sample_id_blinded |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 |  | picnic | U0002 | 2 | 0 | PAR | The family brought food to the park. | 1 | 1 | 1 | 1 |
| 1 |  | picnic | U0003 | 3 | 0 | PAR | The little girl [/] the little girl pours juice. | 0 | 1 | 0 | 1 |
| 1 |  | picnic | U0004 | 4 | 0 | PAR | Then they share sandwiches. | 1 | 0 | 0 | 1 |
| 1 |  | picnic | U0006 | 6 | 0 | PAR | Yes, the dog waits beside them. | 1 | 0 | 0 | 1 |
| 1 |  | picnic | U0007 | 7 | 0 | PAR | The day is quiet. | 0 | 1 | 0 | 1 |
| 3 |  | picnic | U0002 | 2 | 0 | PAR | A picnic. | 1 | 1 | 1 | 2 |
| 3 |  | picnic | U0003 | 3 | 0 | PAR | The dad is opening the basket. | 1 | 0 | 0 | 2 |
| 3 |  | picnic | U0004 | 4 | 0 | PAR | The dog wants food! | 0 | 0 | 0 | 2 |

`expected_outputs/cus_module/cus_analyze/cu_coding_by_sample_long.xlsx`

| coder | paradigm | sv_col | rel_col | cu_col | no_utt | p_sv | m_sv | perc_sv | miss_sv | p_rel | m_rel | perc_rel | miss_rel | cu | perc_cu | miss_cu | sv_rel_inconsistent | sample_id_blinded |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| primary | base | sv | rel | cu | 5 | 3 | 2 | 60 | 0 | 3 | 2 | 60 | 0 | 1 | 20 | 0 | 0 | 1 |
| primary | base | sv | rel | cu | 5 | 4 | 1 | 80 | 0 | 3 | 2 | 60 | 0 | 3 | 60 | 0 | 0 | 2 |
| primary | base | sv | rel | cu | 5 | 3 | 2 | 60 | 0 | 2 | 3 | 40 | 0 | 1 | 20 | 0 | 0 | 3 |

`expected_outputs/cus_module/cus_analyze/cu_coding_by_sample.xlsx`

| no_utt_primary_base | p_sv_primary_base | m_sv_primary_base | perc_sv_primary_base | miss_sv_primary_base | p_rel_primary_base | m_rel_primary_base | perc_rel_primary_base | miss_rel_primary_base | cu_primary_base | perc_cu_primary_base | miss_cu_primary_base | sv_rel_inconsistent_primary_base | sample_id_blinded |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 3 | 2 | 60 | 0 | 3 | 2 | 60 | 0 | 1 | 20 | 0 | 0 | 1 |
| 5 | 4 | 1 | 80 | 0 | 3 | 2 | 60 | 0 | 3 | 60 | 0 | 0 | 2 |
| 5 | 3 | 2 | 60 | 0 | 2 | 3 | 40 | 0 | 1 | 20 | 0 | 0 | 3 |

## Notes

The preview uses synthetic filled coding values generated from the packaged example specs.
