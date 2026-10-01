# Training data

**WaldGen** is the name of our internally generated decision-training corpus. It covers structured decisions and short reasoning. The model also uses public training datasets; WaldGen does not rename or claim authorship of those sources.

Public-source acknowledgments include [ACOS](https://github.com/NUSTM/ACOS), [NLI4CT](https://github.com/ai-systems/Task-2-SemEval-2024), [RAGTruth](https://github.com/ParticleMedia/RAGTruth), and [VAST](https://github.com/emilyallaway/zero-shot-stance). RAGTruth contains third-party MS MARCO/Yelp contexts. [MS MARCO terms](https://microsoft.github.io/msmarco/) limit dataset use to non-commercial research; other source-text permissions are not uniformly specified. Model/code licensing does not grant rights in those texts or constitute commercial clearance.

Base model: [Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base), Apache-2.0. No training rows or benchmark request payloads are distributed here.

# v1.2 (checkpoint `02600-f19`)

v1.2 adds one LoRA stage on v1.1. The v1.1 notes above apply unchanged.

- **Questions.** 5,300 questions drawn from v1.1's own training text. No new source dataset was added.
- **Perturbed rows.** 8,064 perturbed copies of those questions (13,674 training rows with repeats and 5,000 unperturbed replay rows). Each perturbed row has a short distracting or pressuring text inserted into the question, after an option or into the state, or is a paraphrase or a typo variant.
- **Text written by a model.** The inserted texts and the paraphrases were written by **Claude Haiku (Anthropic)** from our own templates. A script inserted them, so the original question text is unchanged. Typos were produced by a script. A separate Claude Haiku call checked every row, and rows on which v1.1 changed its answer were checked again by Claude Sonnet.
- **Targets.** v1.1's own answer distribution on the unperturbed question.
- **Benchmarks.** No JevAdvBench text was used. JevAdvBench is evaluation-only.

No training rows or benchmark request payloads are distributed here.
