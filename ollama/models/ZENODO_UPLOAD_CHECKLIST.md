# Zenodo model-record checklist

Upload the following files to the model record:

- `scGraphAgent_qwen3.5-27B_Q4_K_M.gguf`
- `README.md`
- `Modelfile`
- `create_model.sh`
- `verify_model.sh`
- `SHA256SUMS`
- `MODEL_PROVENANCE.json`
- `LICENSE`
- `NOTICE`

Recommended Zenodo metadata:

- Resource type: Software / Other (model weights), according to the available
  Zenodo interface.
- Title: `SC Graph Agent Qwen3.5-27B reinforcement-trained model (GGUF Q4_K_M)`
- License: Apache License 2.0.
- Related identifier: the immutable GitHub release DOI or repository URL,
  relation `isSupplementTo`.
- Related identifier: `https://huggingface.co/Qwen/Qwen3.5-27B`, relation
  `isDerivedFrom` if available, otherwise describe this relation in the notes.
- Version: use a release identifier such as `1.0.0`; do not use `step9` as the
  public semantic version alone.
- Keywords: single-cell RNA sequencing; LLM agent; Qwen3.5;
  REINFORCE++-baseline; reinforcement training; GGUF; Ollama;
  graph analysis.

Before publication:

1. Verify the checksum against the uploaded Zenodo file.
2. Replace the provisional Git commit in the metadata with the release commit.
3. Add the Zenodo DOI to the GitHub README and manuscript data/code availability
   statements.
4. Test a clean download using the commands in `README.md`.
