import runpod
import numpy as np
import re
from bertalign import Bertalign, model
from typing import Dict, Any
import asyncio
import torch

print("Preloading SentenceTransformer model...")
_ = model.model.encode("warmup")  # Load model weights into memory
print("Model loaded successfully!")

if torch.cuda.is_available():
    torch.cuda.init()
    print(f"CUDA initialized. Using {torch.cuda.get_device_name(0)}")
else:
    print("Warning: Using CPU-only mode!")

def split_into_sentences(text):
    return re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())

def convert_numpy_types(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

async def process_alignment(src, tgt):
    src_sents = split_into_sentences(src)
    aligner = Bertalign(src, tgt, is_split=False)
    aligner.align_sents()
    
    alignments = []
    for src_idx, tgt_idx in aligner.result:
        alignment = {
            'source': [aligner.src_sents[i] for i in (src_idx if isinstance(src_idx, (list, tuple)) else [src_idx])],
            'target': [aligner.tgt_sents[i] for i in (tgt_idx if isinstance(tgt_idx, (list, tuple)) else [tgt_idx])]
        }
        alignments.append(alignment)
    
    return {
        'alignments': convert_numpy_types(alignments),
        'source_sentences': len(src_sents),
        'target_sentences': len(aligner.tgt_sents)
    }

async def handler(event):
    try:
        input_data = event["input"]
        src = input_data.get("src", "").strip()
        tgt = input_data.get("tgt", "").strip()
        
        if not src or not tgt:
            return {
                "error": "Both source and target texts are required",
                "status": "FAILED"
            }
        
        print(f"Processing alignment request...")
        print(f"Source text length: {len(src)}")
        print(f"Target text length: {len(tgt)}")
        
        try:
            result = await asyncio.wait_for(
                process_alignment(src, tgt),
                timeout=600  # 10 minute timeout
            )
            print("Alignment completed successfully")
            
            return {
                "status": "COMPLETED",
                "data": result
            }
            
        except asyncio.TimeoutError:
            print("Alignment timed out")
            return {
                "status": "FAILED",
                "error": "Processing timeout exceeded"
            }
            
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return {
            "status": "FAILED",
            "error": str(e)
        }

runpod.serverless.start({"handler": handler})