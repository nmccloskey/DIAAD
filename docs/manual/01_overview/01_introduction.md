# Introduction to DIAAD

DIAAD, the Database-oriented, Integrative Architecture for Analyzing Discourse, is an open-source system for organizing and analyzing discourse data. It is designed for projects where transcript content, metadata, manual coding, automated first passes, reliability checks, blinding, and analysis outputs need to remain connected across many files and repeated runs.

Most DIAAD workflows are transcript based. In those workflows, the central early step is converting CHAT-formatted transcripts into transcript tables: Excel workbooks that separate sample-level metadata from utterance-level transcript content while preserving stable identifiers. Those tables become the common structure with which users can generate coding files, select reliability subsets, calculate rates, analyze target vocabulary coverage, export revised CHAT files, and combine outputs across modules.

DIAAD also includes tools that are not strictly downstream of transcript tabularization. The templates module can prepare generic coding workbooks for paradigms DIAAD does not directly analyze. The Digital Conversational Turns module supports transcriptionless or pre-transcription coding of turn sequences. The examples module generates synthetic example projects and generated Example I/O documentation. These pieces share the same general design principle: keep data, identifiers, configuration, and output files organized enough that a project can be audited and rerun.

## What DIAAD Does

DIAAD supports several recurring tasks in discourse analysis:

- Converting CHAT transcripts into sample and utterance tables.
- Selecting, evaluating, and reselecting transcription reliability samples.
- Generating manual coding files for Complete Utterances, word counts, POWERS, Digital Conversational Turns, and custom templates.
- Supporting automated first passes for selected coding tasks, with human review expected.
- Evaluating manual coding reliability and preparing replacement reliability subsets when needed.
- Calculating per-minute rates from user-entered speaking-time values in seconds.
- Analyzing target vocabulary coverage with built-in CoreLex-style resources or custom lexicons.
- Blinding and decoding identifiers for manual coding, analysis, or shared outputs.
- Writing logs, resolved configuration files, manifests, and other run artifacts for reproducibility.
- Generating runnable synthetic example packages and generated Example I/O pages.

The program is modular, but the modules are meant to fit together. A user can run one command at a time, chain multiple commands in a single CLI invocation, or use the web app to select commands and download outputs. The manual therefore presents DIAAD both as a set of commands and as a database-oriented workflow scaffold.

## Ways to Use DIAAD

DIAAD can be used in two main ways.

The command-line interface is the more flexible option. It is recommended for sensitive data, large projects, repeated workflows, custom configuration, and users who need the strongest control over local files. CLI runs use a project input directory and output directory, create timestamped outputs, and write run artifacts that help document what happened.

The web app is the more accessible option. It lets users build or upload configuration files, upload input files, choose commands, run DIAAD in a temporary workspace, and download a result ZIP. Because web use requires uploading data to a hosted service, deidentified data are strongly recommended. For highly sensitive, identifiable, or difficult-to-deidentify clinical discourse, local CLI use is usually the safer default.

## Data Privacy and Blinding

DIAAD's blinding tools can replace configured identifier columns with blind codes and later decode them when analysis outputs need to be reconnected to canonical sample identifiers. This can support coder masking and cleaner statistical exports, but software-level blinding is not the same as full de-identification.

For example, a team may deidentify file names and blind sample identifiers before manual coding. If the same people who collected the conversations also analyze them, however, their memory of the conversations or familiarity with participants may still compromise practical blinding. Similarly, replacing names in transcript text does not necessarily remove every sensitive contextual clue. DIAAD can support privacy-conscious workflows, but users remain responsible for deciding whether their data are appropriate for the web app, local processing, sharing, or publication.

## How to Read This Manual

This manual is organized around documentation objects:

- Features explain cross-cutting ideas that matter early, such as transcript tabularization, exact file matching, and generated Example I/O.
- Modules explain major parts of DIAAD, such as Transcripts, Templates, Complete Utterances, Word Counting, POWERS, Target Vocabulary Coverage, Digital Conversational Turns, Blinding, and Examples.
- Commands explain how to run specific DIAAD operations.
- Functionalities explain behavior that spans modules, such as configuration, reliability, rate calculation, revision handling, and provenance.

Generated Example I/O pages are maintained separately from authored manual prose. They are built from synthetic example data and show runnable project structures, representative inputs, and output previews. Authored manual pages explain what to do and why; generated Example I/O pages show concrete files that users can reproduce.

## Scope and Responsibility

DIAAD aims to make discourse-analysis workflows more efficient, systematic, and reproducible. It does not remove the need for methodological judgment. Manual coding protocols still need clear rules, reliability review, and project-specific documentation. Automated first passes still need human inspection. Custom target vocabulary resources should be interpreted cautiously unless independently validated. Configuration choices should be recorded and reviewed as part of a study's analytic method.

The codebase is the ground source of truth for operational details. When this manual explains syntax, file names, settings, and outputs, it should be understood as documentation of the current DIAAD implementation rather than a substitute for project-specific research protocols.

## Acknowledgments

DIAAD builds on and integrates functionality from two excellent open-source resources:

- [**batchalign2**](https://github.com/TalkBank/batchalign2) – Developed by the TalkBank team, Batchalign provides a robust backend for automatic speech recognition (ASR). DIAAD was designed to function downstream of this system, leveraging its debulletized `.cha` files as input. This integration allows researchers to significantly expedite batch transcription, which without an ASR springboard might bottleneck discourse analysis.

> Liu H, MacWhinney B, Fromm D, Lanzi A. *Automation of Language Sample Analysis*. J Speech Lang Hear Res. 2023 Jul 12;66(7):2421-2433. doi: 10.1044/2023_JSLHR-22-00642. Epub 2023 Jun 22. PMID: 37348510; PMCID: PMC10555460.

- [**coreLexicon**](https://github.com/rbcavanaugh/coreLexicon) – A web-based interface for Core Lexicon analysis developed by Rob Cavanaugh, et al (2021). DIAAD's target vocabulary coverage workflow uses built-in CoreLex-style resources and accesses associated normative data using resource-declared Google Sheet URLs to calculate optional percentiles.

  - **Inspiration & overlap:** DIAAD retains the accuracy, efficiency, and percentile concepts from the original web app while exporting batch-friendly summary and long-format detail tables.
  - **Enhancements:** DIAAD supports batch analysis of uploaded/input tabular data and reports normalized target vocabulary coverage for aggregate comparisons across tasks.
  - **Recommended use cases:** The original web app remains an excellent choice for small numbers of individualized reports, while DIAAD's workflow focuses on higher-throughput downstream statistical workflows.

> Cavanaugh, R., Dalton, S. G., & Richardson, J. (2021). coreLexicon: *An open-source web-app for scoring core lexicon analysis*. R package version 0.0.1.0000. https://github.com/aphasia-apps/coreLexicon
