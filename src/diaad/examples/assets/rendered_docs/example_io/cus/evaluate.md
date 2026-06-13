---
object_type: command
object_types:
- command
command_id: cus.evaluate
canonical_command: cus evaluate
module_id: cus
view: example_io
title: CU Reliability Evaluation Example
slot: examples
---

# CU Reliability Evaluation Example

This example demonstrates how `diaad cus evaluate` compares primary CU coding with a synthetic reliability workbook.

## Command

```bash
diaad cus evaluate --config config
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
        cu_reliability_coding.xlsx
    output/
      diaad_YYMMDD_HHMM/
        cu_reliability/
          cu_reliability_coding_by_utterance.xlsx
          cu_reliability_coding_by_sample.xlsx
          cu_reliability_coding_report.txt
        logs/
          diaad_YYMMDD_HHMM.log
          diaad_YYMMDD_HHMM_metadata.json
```

## Basic Config

```yaml
input_dir: diaad_data/input
output_dir: diaad_data/output
```

## Input Snippet

The command reads `diaad_data/input/cu_coding/cu_coding.xlsx` and `diaad_data/input/cu_coding/cu_reliability_coding.xlsx`.

## Output Preview

`diaad_data/output/diaad_YYMMDD_HHMM/cu_reliability/cu_reliability_coding_by_utterance.xlsx`

| utterance_id | sample_id | c2_sv | c2_rel | c3_sv | c3_rel | c2_cu | c3_cu | agmt_sv | agmt_rel | agmt_cu |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| U0001 | 1 |  |  |  |  |  |  | 1 | 1 | 1 |
| U0001 | 1 |  |  |  |  |  |  | 1 | 1 | 1 |
| U0002 | 1 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 0.0 | 1 | 0 | 0 |
| U0002 | 1 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 0.0 | 1 | 0 | 0 |
| U0003 | 1 | 0.0 | 1.0 | 0.0 | 1.0 | 0.0 | 0.0 | 1 | 1 | 1 |
| U0003 | 1 | 0.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 1 |
| U0004 | 1 | 1.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 1 | 1 | 1 |
| U0004 | 1 | 1.0 | 0.0 |  |  | 0.0 |  | 0 | 0 | 0 |

`diaad_data/output/diaad_YYMMDD_HHMM/cu_reliability/cu_reliability_coding_by_sample.xlsx`

| sample_id | num_utterances2 | plus_sv2 | minus_sv2 | plus_rel2 | minus_rel2 | plus_cu2 | perc_cu2 | num_utterances3 | plus_sv3 | minus_sv3 | plus_rel3 | minus_rel3 | plus_cu3 | perc_cu3 | total_agmt_sv | perc_agmt_sv | total_agmt_rel | perc_agmt_rel | total_agmt_cu | perc_agmt_cu | sample_agmt_sv | sample_agmt_rel | sample_agmt_cu |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 10.0 | 6.0 | 4.0 | 6.0 | 4.0 | 2.0 | 20.0 | 10.0 | 7.0 | 3.0 | 4.0 | 6.0 | 2.0 | 20.0 | 10.0 | 71.429 | 8.0 | 57.143 | 8.0 | 57.143 | 0.0 | 0.0 | 0.0 |
| 2.0 | 10.0 | 8.0 | 2.0 | 6.0 | 4.0 | 6.0 | 60.0 | 10.0 | 7.0 | 3.0 | 4.0 | 6.0 | 2.0 | 20.0 | 9.0 | 64.286 | 8.0 | 57.143 | 8.0 | 57.143 | 0.0 | 0.0 | 0.0 |
| 3.0 | 10.0 | 6.0 | 4.0 | 4.0 | 6.0 | 2.0 | 20.0 | 10.0 | 7.0 | 3.0 | 4.0 | 6.0 | 2.0 | 20.0 | 5.0 | 35.714 | 7.0 | 50.0 | 9.0 | 64.286 | 0.0 | 0.0 | 0.0 |

`diaad_data/output/diaad_YYMMDD_HHMM/cu_reliability/cu_reliability_coding_report.txt`

```text
CU Reliability Coding Report
Comparison mode: primary_vs_reliability

Coverage in primary coding file
--------------------------------
Samples represented: 3/3 (100.0%)
Utterances represented: 21/21 (100.0%)

Primary reliability metrics
---------------------------
```

## Notes

The reliability coding values are synthetic and intentionally small. They are meant to show file shape and summary fields, not benchmark agreement.
