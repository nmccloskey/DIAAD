# DIAAD Example I/O Manual

The example I/O manual shows small, runnable DIAAD workflows alongside their inputs and outputs.

Runnable files are generated locally under `example_files/`, so the repository and installed package do not carry generated workbooks or CHAT copies. The manual-style markdown is packaged with DIAAD under `diaad.examples.assets.rendered_docs.example_io` for use by the webapp, manual renderer, or other documentation tools.

Command pages show minimal user project structures: the config files a user needs, the required inputs for that one command, and the output files created in a timestamped DIAAD run directory. The local generated example project is a fuller teaching fixture because it contains inputs and expected outputs for several commands at once.

Generated example projects may include a `README.md` for navigation, but a README is not required in a user's DIAAD project.

Synthetic data are defined in packaged YAML specs. Some markdown pages are authored directly, and others include tables, directory trees, and snippets rendered from those specs or from generated example files.

All example data are synthetic. They are not human-subjects data, participant records, clinical documentation, or de-identified real transcripts.

## Generated Example Files

```
example_files/
  synthetic_project/
    README.md
    config/
      project.yaml
      advanced.yaml
    input/
      chat/
        P1_picnic_pre.cha
        P2_picnic_pre.cha
        P1_picnic_post.cha
        reliability/
          P1_picnic_pre.cha
          P2_picnic_pre.cha
      transcription_reliability_selection/
        transcription_reliability_samples.xlsx
    expected_outputs/
      transcripts_module/
        transcripts_tabularize/
          transcript_table.xlsx
        transcripts_select/
          transcription_reliability_samples.xlsx
        transcripts_evaluate/
          transcription_reliability_evaluation.xlsx
          transcription_reliability_report.txt
        transcripts_reselect/
          reselected_transcription_reliability/
            reselected_transcription_reliability_samples.xlsx
      templates_module/
        templates_utterances/
          utterance_coding_template.xlsx
          utterance_reliability_template.xlsx
          utterance_template_codebook.xlsx
        templates_samples/
          sample_coding_template.xlsx
          sample_reliability_template.xlsx
          sample_template_codebook.xlsx
        templates_times/
          speaking_times.xlsx
```
