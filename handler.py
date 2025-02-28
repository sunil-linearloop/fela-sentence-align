import runpod
import numpy as np
import re
from bertalign import Bertalign, model
from typing import Dict, Any
import torch
import os
import concurrent.futures

# Set environment variables to ensure proper behavior
os.environ["PYTHONPATH"] = "/app"
os.environ["TRANSFORMERS_CACHE"] = "/app/models"
os.environ["HF_HOME"] = "/app/models"

# Print debug info during container startup
print("Preloading SentenceTransformer model...")
try:
    # Force model to load by encoding a dummy string
    _ = model.model.encode("warmup")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Warning: Error during model preloading: {str(e)}")
    print("Will attempt to load model during request processing")

# Check CUDA availability
if torch.cuda.is_available():
    try:
        torch.cuda.init()
        print(f"CUDA initialized. Using {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"Warning: CUDA initialization error: {str(e)}")
        print("Will attempt to run without CUDA")
else:
    print("Warning: Using CPU-only mode!")

def split_into_sentences(text):
    """Split text into sentences more accurately."""
    return re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())

def convert_numpy_types(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization."""
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

def process_alignment(src, tgt):
    """Process text alignment using Bertalign (synchronous version)."""
    src_sents = split_into_sentences(src)
    tgt_sents = split_into_sentences(tgt)
    
    print(f"Source text: {len(src_sents)} sentences")
    print(f"Target text: {len(tgt_sents)} sentences")
    
    aligner = Bertalign(src, tgt, is_split=False)
    aligner.align_sents()
    
    alignments = []
    for src_idx, tgt_idx in aligner.result:
        # Process source sentences
        if isinstance(src_idx, (list, tuple, np.ndarray)):
            source_texts = [aligner.src_sents[i] for i in src_idx]
        else:
            source_texts = [aligner.src_sents[src_idx]]
            
        # Process target sentences
        if isinstance(tgt_idx, (list, tuple, np.ndarray)):
            target_texts = [aligner.tgt_sents[i] for i in tgt_idx]
        else:
            target_texts = [aligner.tgt_sents[tgt_idx]]
            
        alignment = {
            'source': source_texts,
            'target': target_texts
        }
        alignments.append(alignment)
    
    # Prepare detailed result
    result = {
        'alignments': convert_numpy_types(alignments),
        'source_sentences': len(src_sents),
        'target_sentences': len(tgt_sents),
        'total_alignments': len(alignments)
    }
    
    return result

def handler(event):
    """
    RunPod serverless handler function to process sentence alignment requests.
    Input format:
    {
        "input": {
            "src": "Source language text",
            "tgt": "Target language text"
        }
    }
    """
    try:
        # Extract input data
        input_data = event.get("input", {})
        src = input_data.get("src", "").strip()
        tgt = input_data.get("tgt", "").strip()
        
        # Validate input
        if not src or not tgt:
            return {
                "error": "Both source and target texts are required",
                "status": "FAILED"
            }
        
        print(f"Processing alignment request...")
        print(f"Source text length: {len(src)}")
        print(f"Target text length: {len(tgt)}")
        
        # Process the alignment using executor to handle timeouts without event loop conflicts
        with concurrent.futures.ThreadPoolExecutor() as executor:
            try:
                # Run the alignment process in a separate thread with a timeout
                future = executor.submit(process_alignment, src, tgt)
                result = future.result(timeout=600)  # 10 minute timeout
                
                print("Alignment completed successfully")
                
                return {
                    "status": "COMPLETED",
                    "output": result
                }
                
            except concurrent.futures.TimeoutError:
                print("Alignment timed out")
                return {
                    "status": "FAILED",
                    "error": "Processing timeout exceeded"
                }
            
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Error: {str(e)}")
        print(error_traceback)
        
        return {
            "status": "FAILED",
            "error": f"{type(e).__name__}: {str(e)}"
        }

# Start the RunPod serverless worker
runpod.serverless.start({"handler": handler})