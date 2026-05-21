# Conversation Turns Analysis Example

This example demonstrates how `diaad turns analyze` summarizes digital conversation-turn strings across speakers, bins, sessions, and groups.

## Command

```bash
diaad turns analyze --config config
```

## Project Files

```
your_project/
  config/
    project.yaml
    advanced.yaml
  diaad_data/
    input/
      conversation_turns/
        conversation_turns_template.xlsx
    output/
      diaad_YYMMDD_HHMM/
        conversation_turns_template_analysis.xlsx
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

`diaad_data/input/conversation_turns/conversation_turns_template.xlsx`

| sample_id | session | bin | turns |
| --- | --- | --- | --- |
| S001 | visit_1 | bin_1 | 0.1..23.0.12 |
| S001 | visit_1 | bin_2 | 1.0..32.10. |
| S001 | visit_2 | bin_1 | 0.12.3..01 |
| S001 | visit_2 | bin_2 | 2.30.1..20 |
| S002 | visit_1 | bin_1 | 0.13..20.1 |
| S002 | visit_1 | bin_2 | 3.0..12.30 |

## Output Preview

`expected_outputs/turns_module/turns_analyze/conversation_turns_template_analysis.xlsx`

### Sheet: bin_level_turns

| group | session | speaker | bin | turns | mark1 | mark2 | proportion_of_bin_turns | prop_mark1 | prop_mark2 | proportion_of_bin_mark1 | proportion_of_bin_mark2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S001 | visit_1 | 3 | bin_1 | 1 | 1 | 0 | 0.1428571428571428 | 1.0 | 0.0 | 0.3333333333333333 | 0 |
| S001 | visit_1 | 1 | bin_1 | 2 | 0 | 1 | 0.2857142857142857 | 0.0 | 0.5 | 0.0 | 1 |
| S001 | visit_1 | 0 | bin_1 | 2 | 2 | 0 | 0.2857142857142857 | 1.0 | 0.0 | 0.6666666666666666 | 0 |
| S001 | visit_1 | 2 | bin_1 | 2 | 0 | 0 | 0.2857142857142857 | 0.0 | 0.0 | 0.0 | 0 |
| S001 | visit_1 | 3 | bin_2 | 1 | 0 | 0 | 0.1666666666666667 | 0.0 | 0.0 | 0.0 | 0 |
| S001 | visit_1 | 1 | bin_2 | 2 | 1 | 0 | 0.3333333333333333 | 0.5 | 0.0 | 0.3333333333333333 | 0 |
| S001 | visit_1 | 0 | bin_2 | 2 | 1 | 1 | 0.3333333333333333 | 0.5 | 0.5 | 0.3333333333333333 | 1 |
| S001 | visit_1 | 2 | bin_2 | 1 | 1 | 0 | 0.1666666666666667 | 1.0 | 0.0 | 0.3333333333333333 | 0 |

### Sheet: participation_level_turns

| group | session | speaker | total_turns | total_mark1 | total_mark2 | proportion_of_session_turns | prop_mark1 | prop_mark2 | mean_turns | std_turns | var_turns | min_turns | max_turns | mean_mark1 | std_mark1 | var_mark1 | mean_mark2 | std_mark2 | var_mark2 | cv_turns | avg_change_turns |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S001 | visit_1 | 0 | 4 | 3 | 1 | 0.3076923076923077 | 0.75 | 0.25 | 2.0 | 0.0 | 0.0 | 2 | 2 | 1.5 | 0.7071067811865476 | 0.5 | 0.5 | 0.7071067811865476 | 0.5 | 0.0 | 0 |
| S001 | visit_1 | 1 | 4 | 1 | 1 | 0.3076923076923077 | 0.25 | 0.25 | 2.0 | 0.0 | 0.0 | 2 | 2 | 0.5 | 0.7071067811865476 | 0.5 | 0.5 | 0.7071067811865476 | 0.5 | 0.0 | 0 |
| S001 | visit_1 | 2 | 3 | 1 | 0 | 0.2307692307692308 | 0.3333333333333333 | 0.0 | 1.5 | 0.7071067811865476 | 0.5 | 1 | 2 | 0.5 | 0.7071067811865476 | 0.5 | 0.0 | 0.0 | 0.0 | 0.4714045207910317 | 1 |
| S001 | visit_1 | 3 | 2 | 1 | 0 | 0.1538461538461539 | 0.5 | 0.0 | 1.0 | 0.0 | 0.0 | 1 | 1 | 0.5 | 0.7071067811865476 | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| S001 | visit_2 | 0 | 4 | 2 | 0 | 0.3333333333333333 | 0.5 | 0.0 | 2.0 | 0.0 | 0.0 | 2 | 2 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| S001 | visit_2 | 1 | 3 | 0 | 1 | 0.25 | 0.0 | 0.3333333333333333 | 1.5 | 0.7071067811865476 | 0.5 | 1 | 2 | 0.0 | 0.0 | 0.0 | 0.5 | 0.7071067811865476 | 0.5 | 0.4714045207910317 | 1 |
| S001 | visit_2 | 2 | 3 | 2 | 0 | 0.25 | 0.6666666666666666 | 0.0 | 1.5 | 0.7071067811865476 | 0.5 | 1 | 2 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.4714045207910317 | 1 |
| S001 | visit_2 | 3 | 2 | 0 | 1 | 0.1666666666666667 | 0.0 | 0.5 | 1.0 | 0.0 | 0.0 | 1 | 1 | 0.0 | 0.0 | 0.0 | 0.5 | 0.7071067811865476 | 0.5 | 0.0 | 0 |

### Sheet: session_level_summary

| session | group | total_turns | total_mark1 | total_mark2 | turn_entropy | clinician_participant_ratio | prop_mark1 | prop_mark2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| visit_1 | S001 | 13 | 6 | 2 | 1.351681194685895 | 0.4444444444444444 | 0.4615384615384616 | 0.1538461538461539 |
| visit_1 | S002 | 12 | 4 | 2 | 1.357977854987324 | 0.5 | 0.3333333333333333 | 0.1666666666666667 |
| visit_2 | S001 | 12 | 4 | 2 | 1.357977854987324 | 0.5 | 0.3333333333333333 | 0.1666666666666667 |

### Sheet: speaker_level_turns

| group | speaker | total_turns | mark1 | mark2 | unique_sessions | bins_appeared_in | prop_mark1 | prop_mark2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S001 | 0 | 8 | 5 | 1 | 2 | 4 | 0.625 | 0.125 |
| S001 | 1 | 7 | 1 | 2 | 2 | 4 | 0.1428571428571428 | 0.2857142857142857 |
| S001 | 2 | 6 | 3 | 0 | 2 | 4 | 0.5 | 0.0 |
| S001 | 3 | 4 | 1 | 1 | 2 | 4 | 0.25 | 0.25 |
| S002 | 0 | 4 | 2 | 1 | 1 | 2 | 0.5 | 0.25 |
| S002 | 1 | 3 | 0 | 0 | 1 | 2 | 0.0 | 0.0 |
| S002 | 2 | 2 | 1 | 0 | 1 | 2 | 0.5 | 0.0 |
| S002 | 3 | 3 | 1 | 1 | 1 | 2 | 0.3333333333333333 | 0.3333333333333333 |

### Sheet: group_level_summary

| group | total_turns | total_mark1 | total_mark2 | num_participants | bins_covered | num_sessions | prop_mark1 | prop_mark2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S001 | 25 | 10 | 4 | 4 | 16 | 2 | 0.4 | 0.16 |
| S002 | 12 | 4 | 2 | 4 | 8 | 1 | 0.3333333333333333 | 0.1666666666666667 |

### Sheet: summary_statistics

| level | metric | mean | std | min | max | cv |
| --- | --- | --- | --- | --- | --- | --- |
| session | total_turns | 12.33333333333333 | 0.5773502691896257 | 12.0 | 13.0 | 0.04681218398834803 |
| session | total_mark1 | 4.666666666666667 | 1.154700538379252 | 4.0 | 6.0 | 0.2474358296526968 |
| session | total_mark2 | 2.0 | 0.0 | 2.0 | 2.0 | 0.0 |
| session | turn_entropy | 1.355878968220181 | 0.003635378520025889 | 1.351681194685895 | 1.357977854987324 | 0.002681196924824296 |
| session | clinician_participant_ratio | 0.4814814814814815 | 0.03207501495497923 | 0.4444444444444444 | 0.5 | 0.06661733875264916 |
| session | prop_mark1 | 0.3760683760683761 | 0.07401926528072128 | 0.3333333333333333 | 0.4615384615384616 | 0.1968239554055543 |
| session | prop_mark2 | 0.1623931623931624 | 0.007401926528072115 | 0.1538461538461539 | 0.1666666666666667 | 0.04558028440970724 |
| participation | total_turns | 3.083333333333333 | 0.7929614610987591 | 2.0 | 4.0 | 0.257176690086084 |

### Sheet: speaker_level_ratios

| group | participant_to_participant | participant_to_clinician | clinician_to_participant |
| --- | --- | --- | --- |
| S001 | 1.716666666666667 | 1.283333333333333 | 1 |
| S002 | 1.833333333333333 | 1.166666666666667 | 1 |

### Sheet: speaker_matrix_S001

| Unnamed: 0 | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- |
| 0.0 | 0.0 | 0.8333333333333334 | 0.0 | 0.1666666666666667 |
| 1.0 | 0.3333333333333333 | 0.0 | 0.6666666666666666 | 0.0 |
| 2.0 | 0.2 | 0.2 | 0.0 | 0.6 |
| 3.0 | 0.75 | 0.0 | 0.25 | 0.0 |

### Sheet: speaker_matrix_S002

| Unnamed: 0 | 0 | 1 | 2 | 3 |
| --- | --- | --- | --- | --- |
| 0.0 | 0.0 | 1.0 | 0.0 | 0.0 |
| 1.0 | 0.0 | 0.0 | 0.5 | 0.5 |
| 2.0 | 0.5 | 0.0 | 0.0 | 0.5 |
| 3.0 | 0.6666666666666666 | 0.0 | 0.3333333333333333 | 0.0 |

## Notes

The strings are deliberately tiny but include two sessions, two bins, four speakers, and both dot-marker forms.
