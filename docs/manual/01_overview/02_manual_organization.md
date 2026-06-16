# Manual Organization

This manual is organized around documentation objects and documentation views. The structure follows the IRIDIC documentation ontology, adapted for DIAAD's command-line, web-app, and discourse-analysis workflows.

The main idea is simple: users usually arrive with different questions. One user may need the fastest safe way to run a command. Another may need to understand a coding workflow before adopting it in a study. Another may need implementation details for troubleshooting or reproducibility. Object-oriented documentation keeps those questions close to the part of DIAAD they concern, while separating quick operational guidance from deeper methodological and technical context.

## Documentation Objects

A documentation object is the thing being explained.

| Object type | What it means in this manual | Examples |
|---|---|---|
| Feature | A cross-cutting concept that affects several parts of DIAAD. | Transcript tabularization, exact file name matching, generated Example I/O. |
| Module | A major DIAAD subsystem or domain area. | Transcripts, Templates, Complete Utterances, Word Counting, POWERS, Target Vocabulary Coverage, Digital Conversational Turns, Blinding, Examples. |
| Command | A user-invokable DIAAD operation. | `diaad transcripts tabularize`, `diaad cus analyze`, `diaad powers files`, `diaad examples`. |
| Functionality | Shared behavior that appears across modules or commands. | Configuration, reliability selection and evaluation, blinding, speaking-time rates, metadata extraction, run provenance. |
| Workflow | A multi-step applied path through DIAAD. | CLI first run, transcription-based workflow baseline, monologic narrative analysis, clinician-client POWERS coding. |

These categories intentionally overlap in a few places. For example, blinding is both a module and a broader functionality because it has commands of its own but also affects other coding workflows through `auto_blind`. The examples system is both a command and a workflow because users can run `diaad examples`, but example packages also teach the shape of a DIAAD project.

## Documentation Views

A view is the kind of explanation attached to an object.

| View | Main question | Typical use |
|---|---|---|
| Quickstart | What is the shortest reliable path to begin? | Run the command, identify required inputs, see expected outputs, and know what to read next. |
| Usage Guide | How should I use this in a real project? | Work through settings, branch points, review steps, common problems, and project decisions. |
| Research Context | Why does this matter methodologically? | Understand interpretation limits, reliability, validity, privacy, and discourse-analysis context. |
| Implementation Notes | What does the current implementation actually do? | Check file discovery, source behavior, defaults, output shapes, logs, and troubleshooting details. |

Not every object needs every view. A short feature may work best as one integrated page. A command usually needs a Quickstart, Usage Guide, and Implementation Notes. A module usually needs a Quickstart, Research Context, and Implementation Notes. A workflow usually needs a Quickstart and Usage Guide, with enough Research Context to keep the sequence methodologically grounded.

The goal is not to multiply files for their own sake. A view appears only when it answers a non-redundant question.

## How The Manual Tree Uses Objects

The authored manual is arranged in numbered sections:

```text
docs/manual/
  01_overview/
  02_operation/
  03_features/
  04_modules/
  05_functionalities/
  06_workflows/
  99_references.md
```

The Overview section explains what DIAAD is and how to read the manual. Operation explains installation, configuration, command-line use, web-app use, and testing. Features introduce concepts that users should know before diving into module details.

Modules are the main domain sections. Command pages live under their parent modules because commands are usually easiest to understand in the context of the subsystem they operate on. Functionality pages collect shared behavior that would otherwise be repeated across many command pages. Workflow pages come later because they synthesize several objects into project-level paths.

## Reading Paths

For a first encounter with DIAAD, read:

```text
docs/manual/01_overview/01_introduction.md
docs/manual/02_operation/01_installation.md
docs/manual/02_operation/02_command_line.md
docs/manual/02_operation/04_configuration.md
```

For a task-focused path, start with a Workflow page and follow its Read Next references into modules, commands, and functionality pages.

For a command-focused path, start with the command Quickstart, then move to the Usage Guide if settings or file placement matter, and Implementation Notes if behavior is surprising.

For a methodological path, read the relevant module or workflow Research Context page before treating outputs as study variables.

## Generated Example I/O

DIAAD also produces generated Example I/O material from synthetic example packages. These generated pages are not a replacement for authored manual views. They serve a different purpose: showing concrete input and output files that users can reproduce.

Authored pages explain what an object is, when to use it, and how to interpret it. Generated Example I/O pages show tangible toy data, file structures, and output previews. Both support operational understanding, but they do not need to use the same examples or cover the same explanatory ground.

## Navigation Conventions

Authored pages use semantic manual references such as:

```text
docs/manual/04_modules/01_transcripts/01_quickstart.md
```

rather than ordinary Markdown links for internal navigation. This keeps the manual friendly to later rendering in different contexts, including static pages, generated outlines, PDFs, and the web app.

Most pages end with a Read Next section. Those references are meant to help users move across objects without forcing each page to repeat the same background material.

## Why This Structure Matters

DIAAD is modular software for integrative research workflows. A flat manual would either bury quick operational steps inside long methodological discussion or scatter important methodological cautions across command pages. The object/view structure keeps the manual usable for both purposes.

Quickstarts help users get unstuck quickly. Usage Guides support real project work. Research Context pages keep coding, reliability, privacy, and validity decisions visible. Implementation Notes make the current software behavior auditable. Together, those views help the manual function as both user documentation and a reproducibility aid.
