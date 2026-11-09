# EviRAR

This is the official repository for the paper **Evidence-Guided Retrieval-Augmented Reasoning for Multi-turn Inquiry and Syndrome Differentiation in Traditional Chinese Medicine**.

## Installation

```bash
git clone https://github.com/LeoVinchi774/EviRAR.git
cd EviRAR
python -m pip install -e ".[dev]"
cp .env.example .env
```

Edit `.env` to specify separate service endpoints, model names, and API keys for the doctor, embeddings, and optional patient/assistant simulation.

## Model Deployment Reference

EviRAR does not install a model runtime or include model weights. The research project uses separate models for different roles, for example, local [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) acts as the doctor and performs internal reasoning and self-refinement; DeepSeek-V4-Flash simulates patients/assistants and performs expert scoring in experiments; [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) provides retrieval embeddings.

Install the required inference dependencies on a machine with suitable hardware and drivers, then download the model weights. The following example uses the Hugging Face CLI:

```bash
python -m pip install -U huggingface_hub
hf download Qwen/Qwen3.5-4B --local-dir ./models/Qwen3.5-4B
hf download Qwen/Qwen3-Embedding-4B --local-dir ./models/Qwen3-Embedding-4B
```

### Doctor Model

An OpenAI-compatible chat service can be launched using a vLLM version that supports Qwen3.5-4B. Install vLLM separately in the model-serving environment before running the command below. The following command illustrates the service interface; adjust GPU memory and other settings to the deployment environment:

```bash
vllm serve ./models/Qwen3.5-4B \
  --host 127.0.0.1 \
  --port port \
  --served-model-name qwen3.5-4b \
  --api-key example-doctor-key \
  --trust-remote-code
```

### Embedding Model

In the example below, `serve_qwen3_embedding.py` is an illustrative name for a custom server program. To use this approach, first provide a server script implementing the embedding procedure and API described above. The command only illustrates the expected startup arguments and cannot be executed directly in the current repository. 

```bash
python serve_qwen3_embedding.py \
  --model ./models/Qwen3-Embedding-4B \
  --served-model-name qwen3-embedding \
  --api-key example-embedding-key \
  --host 127.0.0.1 \
  --port port
```

### Environment Configuration

Configure `.env` as follows:

```dotenv
TCM_LLM_API_BASE=http://127.0.0.1:port/v1
TCM_LLM_API_KEY=example-doctor-key
TCM_LLM_MODEL=qwen3.5-4b

TCM_EMBEDDING_API_BASE=http://127.0.0.1:port/v1
TCM_EMBEDDING_API_KEY=example-embedding-key
TCM_EMBEDDING_MODEL=qwen3-embedding

# External model for patient/assistant simulation and expert scoring
TCM_SIMULATOR_API_BASE=https://your-endpoint/compatible-mode/v1
TCM_SIMULATOR_API_KEY=replace-with-your-api-key
TCM_SIMULATOR_MODEL=deepseek-v4-flash-0731
```

`TCM_LLM_*` configures the local doctor model, which also performs EviRAR's internal reasoning and self-refinement.

`TCM_EMBEDDING_*` configures the vector retrieval service. EviRAR uses this service to encode knowledge cards, experience cards, and the current retrieval query as vectors for semantic similarity retrieval.

`TCM_SIMULATOR_*` configures patient/assistant simulation and ATCMD-Bench expert scoring after each consultation. Set the external API endpoint for your service provider. The local ports and served model names should match those in your deployment; use the actual API keys configured for each service.

## Benchmark Dataset

Both Baseline and EviRAR case runs use the publicly released ATCMD-Bench dataset and do not require manually entered symptoms. Consult the paper's [ATCMD-Bench repository](https://github.com/mazai-xiangji/ATCMD-Bench) for dataset information and download the test set from the authors' [Google Drive directory](https://drive.google.com/drive/folders/1EggblI-yPC6QQ63YflPZUpqyl2IediAn?usp=sharing). The authors' code uses `tcmd_eval.json` as the default filename. Place the file under `data/` in the project root:

```text
EviRAR/
└── data/
    └── tcmd_eval.json
```

## Inference Process

Run Baseline on the complete dataset, without retrieval augmentation or self-refinement:

```bash
./scripts/run_baseline.sh --dataset data/tcmd_eval.json --simulated-patient
```

Run EviRAR on the complete dataset, with knowledge and experience retrieval and self-refinement:

```bash
./scripts/run_evirar.sh --dataset data/tcmd_eval.json --simulated-patient
```

To recalculate the score summary from saved per-case results, run:

```bash
python scripts/summarize_results.py results/baseline_XXX.jsonl
```


## Resources and Data Sources

`resources/knowledge/` contains 500 syndrome knowledge cards, and `resources/experience/` contains 500 Markdown consultation-experience cards. See [resources/README.md](resources/README.md) for card fields and usage instructions. The complete resource collections are planned for release upon acceptance of the paper.

## Project Structure

```text
configs/                 Baseline and EviRAR YAML configurations
resources/knowledge/     Syndrome knowledge cards
resources/experience/    Consultation-experience cards
src/evirar/              Retrieval, self-refinement, service clients, and CLI
scripts/                 Installation checks and run scripts
tests/                   Offline tests
```
