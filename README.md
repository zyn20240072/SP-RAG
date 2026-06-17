# SP-RAG

Existing multimodal retrieval-augmented generation systems often rely on structural priors, such as shared document IDs, page indices, or predefined cross-modal links, to connect textual and visual evidence. These shortcuts become unreliable in heterogeneous environments where relevant evidence is distributed across decoupled repositories. We propose SP-RAG, a self-perceiving retrieval framework that formulates multimodal retrieval as an active gap-resolution process: the agent maintains grounded beliefs, identifies epistemic gaps, generates modality-aware search actions, and decides whether to retrieve further, answer directly, or stop with sufficient evidence. To make this behavior learnable, we introduce perception-aligned optimization with cognitive and planning rewards for faithful belief-gap assessment and gap-driven planning. Experiments on four multimodal retrieval benchmarks show that SP-RAG (GRPO) consistently improves open-source VLM backbones over non-retrieval, standard MM-RAG, prompted MM-RAG, and recent agentic retrieval baselines. Under the primary structureagnostic independent setting, Qwen2.5-VL-7B with SP-RAG (GRPO) improves over the standard MultiRAG baseline by 9.47–31.25 accuracy points across applicable benchmarks and retrieval settings. Process-level evaluations show that these gains stem from better evidence sufficiency assessment, gap identification, and retrieval action planning. These results suggest that robust multimodal RAG in heterogeneous environments requires trainable self-perceiving retrieval policies rather than metadata-dependent retrieval pipelines

## Repository Structure

```text
.
├── infer_code/                 # Online inference pipeline
│   ├── main.py                 # Dataset-level runner
│   ├── agent.py                # Iterative retrieval and reasoning agent
│   ├── llm_roles.py            # Validator, cognitive planner, and responder prompts
│   ├── retriever.py            # Chroma retriever wrappers
│   ├── tools.py                # Text/image search interface
│   ├── data_schemas.py         # Pydantic schemas
│   ├── memory_system.py        # Agent memory and evidence selection
│   ├── api_handler.py          # OpenAI-compatible multimodal API calls
│   ├── config.py               # Dataset, model, and path configuration
│   └── vllm_start.sh           # Example vLLM serving command
├── retrieval_part/
│   ├── bgem3_service.py        # BGE-M3 text embedding service
│   ├── bgevl_service.py        # BGE-VL image/text-query embedding service
│   └── prepare/
│       ├── get_ocr.py          # OCR extraction for image corpora
│       ├── pack_cleanocr.py    # OCR cleaning and JSONL packing
│       └── embedding_store_v3.py # Chroma index construction
├── train_code/
│   └── grpo_vrag_v9_vllm.sh    # Example GRPO training script
└── final_images/               # Figures used for paper or project description
```

## Environment

The code was written for Python 3.10+ and GPU-based local services. Install dependencies according to your environment. A typical setup requires:

```bash
pip install openai httpx pillow tqdm numpy requests pydantic json-repair
pip install langchain langchain-community langchain-core chromadb
pip install fastapi uvicorn torch transformers FlagEmbedding paddleocr opencv-python
```

For training, the script expects the `swift` RLHF toolkit, DeepSpeed, FlashAttention, vLLM, and a Qwen2.5-VL-compatible environment.

## Configuration

Before running the project, edit the local configuration files:

1. `infer_code/config.py`
   - Set `ROLE_MODEL_CONFIGS` to your OpenAI-compatible model endpoints.
   - Set `PERSIST_DIR_PRE` to the root directory of your Chroma indices.
   - Update `ALL_DATASET_CONFIGS` with your input JSONL files, output paths, and image roots.

2. `retrieval_part/prepare/embedding_store_v3.py`
   - Set `persist_directory` to the index output directory.
   - Update `JSON_DATASETS` and `IMAGE_FOLDER_DATASETS` to point to your local corpus files.

3. `infer_code/vllm_start.sh`
   - Replace the model path and LoRA checkpoint path with your local checkpoints.
   - Set GPU IDs and vLLM serving parameters for your machine.

4. `train_code/grpo_vrag_v9_vllm.sh`
   - Replace the placeholder model, dataset, and logging values.
   - Set the output directory and distributed training parameters.

## Data Format

Inference expects JSONL files where each line is a question item. The required fields are:

```json
{
  "question_id": "example-id",
  "question": "Question text"
}
```

The runner appends the following fields to each processed item:

```json
{
  "final_answer": "Predicted answer",
  "agent_trace": {
    "execution_trace": [],
    "final_assets_summary": {},
    "final_validations_map": {}
  }
}
```

Chroma indices are expected under:

```text
<PERSIST_DIR_PRE>/<dataset>/text
<PERSIST_DIR_PRE>/<dataset>/image
<PERSIST_DIR_PRE>/<dataset>/table   # optional, used for multimodal QA tables
```

## Build Retrieval Indices

Start the embedding services:

```bash
python retrieval_part/bgem3_service.py
python retrieval_part/bgevl_service.py
```

Prepare OCR text when needed:

```bash
python retrieval_part/prepare/get_ocr.py
python retrieval_part/prepare/pack_cleanocr.py
```

Build Chroma indices:

```bash
python retrieval_part/prepare/embedding_store_v3.py
```

`pack_cleanocr.py` removes empty OCR text files in its current form. Back up raw OCR outputs before running it if those files should be preserved.

## Run Inference

Start the multimodal generation model, for example with vLLM:

```bash
bash infer_code/vllm_start.sh
```

Then run the agent:

```bash
cd infer_code
python main.py
```

The pipeline resumes automatically by reading previously processed `question_id` values from the output JSONL file.

## Retrieval Strategies

Set `RETRIEVAL_STRATEGY` in `infer_code/config.py`:

- `independent`: runs text and image retrieval independently for each planned query.
- `intersect`: performs late fusion and prioritizes assets that are retrieved by both modalities.
- `union_rrf`: performs reciprocal-rank fusion across text and image retrieval results.

## Training

The training script provides an example GRPO configuration:

```bash
bash train_code/grpo_vrag_v9_vllm.sh
```

Replace all placeholder paths and model identifiers before use. The script is intended as a reproducibility reference rather than a plug-and-play launcher.

## Citation

For double-blind review, citation information is intentionally omitted. Add the final citation after the review period.
