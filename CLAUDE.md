# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. IMPORTANT: Once this file has been read or updated, it MUST be loaded at the beginning of any new conversation to ensure awareness of communication requirements, custom tasks, etc.


## Project Overview

InnerPO (Inner Preference Optimization) is a research project that aligns language model internal hidden states rather than just outputs, hypothesizing this leads to better out-of-distribution generalization. The project implements DPO (Direct Preference Optimization) as a baseline and several InnerPO variants with different hidden state transforms (SUPR, ETHER, ORTHO).

## User Preferences and Feedback

**CRITICAL**: The user has requested "radical simplification" and emphasized "Less is more. Be bold". They want to focus ONLY on DPO and InnerPO essentials for their paper.

**Package Structure**: Files should be organized within the `reprpo/` package structure, not in the root directory. The project has `uv` and an editable self-install available.

**Real Transforms**: User specifically noted "your transforms are fake, you should just import them from the reprpo directory". Real transform implementations exist in:
- `reprpo/interventions/transforms/supressed.py` - SupressedHSTransform class
- `reprpo/interventions/transforms/ether.py` - ETHERLinear class with orthogonal transformations  
- `reprpo/interventions/transforms/none.py` - NoneTransforms class

**CLI Issues**: User pointed out "we should run your commands, they give lots of errors" - CLI configuration has tyro errors that need fixing.

**Commands**: Use the pattern `python -m reprpo.cli` for running commands from within the package.

YAML configs in `configs/` provide hardware-specific presets (dev.yaml, llama-3-2-1b_a100.yaml, etc.) but these might need to be refactored to tyro dataclasses that are heirarchical and more flexible.

# Code Style Consistency

Default Mode

    Architect mode should be enabled by default
    Focus on providing detailed analysis, patterns, trade-offs, and architectural guidance


    ALWAYS respect how things are written in the existing project
    DO NOT invent your own approaches or innovations
    STRICTLY follow the existing style of tests, resolvers, functions, and arguments
    Before creating a new file, ALWAYS examine a similar file and follow its style exactly
    If code doesn't include comments, DO NOT add comments
    Use seeded data in tests instead of creating new objects when seeded data exists
    Follow the exact format of error handling, variable naming, and code organization used in similar files
    Never deviate from the established patterns in the codebase

Code Documentation and Comments

When working with code that contains comments or documentation:

    Carefully follow all developer instructions and notes in code comments
    Explicitly confirm that all required steps from comments have been completed
    Automatically execute all mandatory steps mentioned in comments without requiring additional reminders
    Treat any comment marked for "developers" or "all developers" as directly applicable to Claude
    Pay special attention to comments marked as "IMPORTANT", "NOTE", or with similar emphasis

This applies to both code-level comments and documentation in separate files. Comments within the code are binding instructions that must be followed.



## Research Focus

The project specifically measures generalization via distribution shift evaluation using the `open_pref_eval` library. The core research question is whether aligning internal states (InnerPO) generalizes better than aligning outputs (DPO).

Evaluation categorizes results by shift type:
- In-distribution (same as training dataset)
- Moderate shift (similar domains)  
- Domain shift (different domains entirely)
- Quality shift (high/low quality variants)

## Dependencies

Core: torch, transformers, peft, datasets, wandb, tyro, open-pref-eval
Optional: lightning (for better experiment tracking)
Development: ruff, pytest, jupyter

The project uses `uv` for dependency management and `just` for task running.
