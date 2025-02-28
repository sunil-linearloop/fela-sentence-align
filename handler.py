import runpod
from bertalign import Bertalign
from typing import Dict, Any

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle incoming alignment requests
    """
    try:
        # Get input texts from the event
        job_input = event["input"]
        src_text = job_input.get("src", "")
        tgt_text = job_input.get("tgt", "")
        
        # Create aligner instance
        aligner = Bertalign(src_text, tgt_text)
        aligner.align_sents()
        
        # Get alignment results
        alignments = []
        for src_idx, tgt_idx in aligner.result:
            alignment = {
                'source': aligner.src_sents[src_idx] if isinstance(src_idx, int) else [aligner.src_sents[i] for i in src_idx],
                'target': aligner.tgt_sents[tgt_idx] if isinstance(tgt_idx, int) else [aligner.tgt_sents[i] for i in tgt_idx]
            }
            alignments.append(alignment)
            
        return {
            "status": "success",
            "alignments": alignments,
            "source_sentences": len(aligner.src_sents),
            "target_sentences": len(aligner.tgt_sents)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler}) 