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
      cu_coding/
        cu_coding.xlsx
        cu_reliability_coding.xlsx
        cu_blind_codebook.xlsx
      cu_coding_analysis/
        cu_coding_by_sample_long.xlsx
      word_counts/
        word_counting.xlsx
        word_count_reliability.xlsx
        word_count_blind_codebook.xlsx
      word_count_analysis/
        word_counting_by_sample.xlsx
      powers_coding/
        powers_coding.xlsx
        powers_reliability_coding.xlsx
      powers_coding_analysis/
        powers_analysis.xlsx
      target_vocab/
        resources/
          picnic_target_vocab.json
        unblind_utterance_data.xlsx
      target_vocab_analysis/
        target_vocab_data_260101_0000.xlsx
      conversation_turns/
        conversation_turns_template.xlsx
        conversation_turns_reliability_template.xlsx
      speaking_times/
        speaking_times.xlsx
    expected_outputs/
      blinding_module/
        blinding_encode/
          powers_coding_blinded.xlsx
          powers_coding_blinding_diagnostics.xlsx
          blind_codebook.xlsx
        blinding_decode/
          cu_coding_decoded.xlsx
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
      cus_module/
        cus_files/
          cu_coding.xlsx
          cu_reliability_coding.xlsx
          cu_blind_codebook.xlsx
        cus_evaluate/
          cu_reliability_coding_by_utterance.xlsx
          cu_reliability_coding_by_sample.xlsx
          cu_reliability_coding_report.txt
        cus_reselect/
          reselected_cu_reliability_coding.xlsx
        cus_analyze/
          cu_coding_by_utterance.xlsx
          cu_coding_by_sample_long.xlsx
          cu_coding_by_sample.xlsx
        cus_rates/
          cu_coding_rates.xlsx
      words_module/
        words_files/
          word_counting.xlsx
          word_count_reliability.xlsx
          word_count_blind_codebook.xlsx
        words_evaluate/
          word_count_reliability_results.xlsx
          word_count_reliability_report.txt
        words_reselect/
          reselected_word_count_reliability.xlsx
        words_analyze/
          word_counting_by_utterance.xlsx
          word_counting_by_sample.xlsx
        words_rates/
          word_counting_rates.xlsx
      powers_module/
        powers_files/
          powers_coding.xlsx
          powers_reliability_coding.xlsx
        powers_evaluate/
          powers_reliability_results.xlsx
          powers_reliability_report.txt
        powers_reselect/
          reselected_powers_reliability_coding.xlsx
        powers_analyze/
          powers_analysis.xlsx
        powers_rates/
          powers_coding_rates.xlsx
      vocab_module/
        vocab_file/
          target_vocabulary_resource_template.json
        vocab_check/
          target_vocab_resource_check.txt
        vocab_analyze/
          target_vocab_data_260101_0000.xlsx
        vocab_rates/
          target_vocab_rates.xlsx
      turns_module/
        turns_files/
          conversation_turns_template.xlsx
          conversation_turns_reliability_template.xlsx
          conversation_turns_template_codebook.xlsx
        turns_evaluate/
          conversation_turns_reliability_results.xlsx
          conversation_turns_reliability_report.txt
        turns_reselect/
          reselected_conversation_turns_reliability_template.xlsx
        turns_analyze/
          conversation_turns_template_analysis.xlsx
```
