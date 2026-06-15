# Blinding Research Context

Blinding can reduce bias in manual coding workflows by hiding selected identifiers from coders. In DIAAD, blinding means deterministic replacement of configured columns with blind codes and preservation of a codebook for later recovery.

## Blinding Is Not De-Identification

Software blinding is not the same as full de-identification, privacy protection, or practical coder masking. A workbook may hide `sample_id` while transcript content, filenames, contextual details, or staff memory still reveal the participant or session.

For example, a lab may deidentify filenames and blind sample IDs, but if the same staff conducted the conversations and later code them, memory of the interaction may limit the practical value of formal blinding.

## Encode And Decode

Encoding supports a masked coding stage. For manual coding procedures, this is usually the recommended direction of travel: prepare coder-facing files, encode configured identifiers, distribute the blinded files, and preserve the codebook separately.

Decoding supports analysis after coding, when results need to reconnect with canonical identifiers, metadata, transcript tables, or other project files. In most DIAAD workflows, decoding back to original sample identifiers before analysis is safer than asking every downstream analysis command to infer blinded identifiers.

After analysis, users may prefer another encoding step for statistical workflows, exports, or collaboration packages where analysts should not see raw identifiers. That later encoding is a sharing and analysis-design choice, not a substitute for preserving the original codebook and decoded canonical analysis files.

The codebook is therefore a sensitive reproducibility artifact. Losing it can make decoded analysis difficult; sharing it too broadly can defeat the purpose of masking.

## Web And Local Workflows

The web app runs in a temporary workspace and returns outputs as a ZIP, but sensitive transcript data still require project-level judgment. For workflows with identifying or highly sensitive content, local CLI use may be preferable even when software-level blinding is available.

## Review Flag

Privacy, de-identification, web-vs-CLI, and codebook storage guidance should receive human review before publication.
