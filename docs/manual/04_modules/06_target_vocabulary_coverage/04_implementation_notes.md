# Target Vocabulary Coverage Implementation Notes

The Target Vocabulary Coverage module is implemented under `src/diaad/coding/target_vocab/`.

## Resources

Resource loading starts with bundled JSON resources. If `advanced.target_vocabulary_resource_path` is configured, DIAAD loads custom JSON resources as well. A custom resource with the same resource ID as a built-in resource can override the bundled version.

Resources declare required metadata, base forms, variant maps, and optional norm specifications. Runtime fields include reverse variant lookups used during token matching.

## Resource Commands

`vocab file` writes `target_vocabulary_resource_template.json` under `target_vocab/`.

`vocab check` validates active resources and writes `target_vocab_resource_check.txt`. The check reports structure and consistency; it does not establish research validity.

## Analysis

`vocab analyze` currently passes through the CLI transcript-table prerequisite gate. Inside the analysis implementation, it prepares input from preferred unblinded utterance data when available, otherwise from transcript tables. It filters to stimuli present in the active resources, removes excluded speakers when speaker labels are available, reformats text, matches variants to base forms, and writes a timestamped `target_vocab_data_*.xlsx` workbook with `summary` and `details` sheets.

## Rates

`vocab rates` finds `target_vocab_data_*.xlsx` analysis workbooks, reads their `summary` sheets, infers count-like numerator columns, converts speaking time to minutes, and writes `target_vocab_rates.xlsx`.

## Boundaries

The module is resource-driven. Incorrect stimulus labels, missing speaking time, or poorly specified resources can prevent analysis or make outputs difficult to interpret.
