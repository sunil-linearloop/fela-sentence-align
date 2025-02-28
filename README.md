# Fela Sentence Aligner - RunPod Serverless Worker

This is a RunPod serverless worker for aligning sentences between two text inputs in different languages. The service uses Bertalign, a BERT-based sentence aligner that provides high-quality cross-lingual sentence alignment.

## Features

- Cross-lingual sentence alignment using BERT embeddings (LaBSE model)
- Handles multiple alignment types (1-1, 1-many, many-1, many-many)
- GPU acceleration for fast processing
- Supports any language pair that BERT supports
- Deployed as a RunPod serverless worker
- Built with CUDA 12.4 support for optimal GPU performance

## API Usage

### Input Format

Send a POST request to your RunPod endpoint with the following JSON payload:

```json
{
  "input": {
    "src": "Your source language text here. It can be multiple sentences.",
    "tgt": "Your target language text here. It can also be multiple sentences in a different language."
  }
}
```

### Output Format

The response will be in the following format:

```json
{
  "status": "COMPLETED",
  "output": {
    "alignments": [
      {
        "source": ["Source sentence 1"],
        "target": ["Target sentence 1"]
      },
      {
        "source": ["Source sentence 2"],
        "target": ["Target sentence 2, Target sentence 3"]
      },
      ...
    ],
    "source_sentences": 10,
    "target_sentences": 12,
    "total_alignments": 8
  }
}
```

## Local Development

To test the worker locally:

```bash
python handler.py --test_input '{"input": {"src": "Hello world. How are you today?", "tgt": "Hallo Welt. Wie geht es dir heute?"}}'
```

## Limitations

- Maximum input size is limited by the RunPod payload limit (2MB)
- Quality of alignment depends on language pair and text domain

## Environment Variables

- `RUNPOD_DEBUG_LEVEL`: Set to `INFO` for detailed logs
- `RUNPOD_WORKER_PORT`: The port the worker runs on (default: 8000)
- `PYTHONPATH`: Set to `/app` to ensure proper module loading

## Technical Details

- Uses RunPod base image: `runpod/base:0.6.2-cuda12.4.1`
- Python 3.10 (explicitly installed)
- PyTorch >= 2.2.0
- RunPod SDK >= 1.6.0
- Uses Hugging Face Transfer optimization for faster model downloads
- Pre-downloads and caches the LaBSE model during Docker build

## Optimizations

The Docker image includes several optimizations:

1. **HF Transfer** - Uses Rust-based library for faster Hugging Face model downloads
2. **Model Caching** - Pre-downloads the LaBSE model during build to avoid runtime downloads
3. **Robust Error Handling** - Gracefully handles network issues or model loading problems
4. **CUDA Support** - Optimized for CUDA 12.4.1 GPUs for faster processing
5. **Thread-based Processing** - Uses ThreadPoolExecutor instead of asyncio to avoid event loop conflicts

