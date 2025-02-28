from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bertalign import Bertalign
from typing import Dict, Any, List, Optional
import numpy as np
import re
import uvicorn

app = FastAPI(
    title="Fela Sentence Aligner",
    version="1.0"
)

class AlignmentRequest(BaseModel):
    src: str
    tgt: str

class AlignmentResponse(BaseModel):
    status: str
    alignments: List[Dict[str, Any]]
    total_alignments: int
    source_sentences: int
    target_sentences: int

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    return obj

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    return sentences

@app.post("/align", response_model=AlignmentResponse)
def align_texts(request: AlignmentRequest):
    try:
        src_text = request.src.strip()
        tgt_text = request.tgt.strip()
        src_sentences = split_into_sentences(src_text)
        aligner = Bertalign("\n".join(src_sentences), tgt_text)
        aligner.align_sents()
        alignments = []
        current_src_idx = 0
        
        for src_idx, tgt_idx in aligner.result:
            try:
                while current_src_idx < (src_idx if isinstance(src_idx, (int, np.integer)) else src_idx[0]):
                    alignments.append({
                        'source': src_sentences[current_src_idx],
                        'target': None,
                        'source_idx': current_src_idx,
                        'target_idx': current_src_idx
                    })
                    current_src_idx += 1
                if isinstance(src_idx, (int, np.integer)) and isinstance(tgt_idx, (int, np.integer)):
                    alignment = {
                        'source': src_sentences[src_idx],
                        'target': aligner.tgt_sents[tgt_idx],
                        'source_idx': current_src_idx,
                        'target_idx': current_src_idx
                    }
                else:
                    src_text_list = [src_sentences[i] for i in (src_idx if isinstance(src_idx, (list, tuple, np.ndarray)) else [src_idx])]
                    for i, src_text in enumerate(src_text_list):
                        target_text = aligner.tgt_sents[tgt_idx[0]] if (i == 0 and isinstance(tgt_idx, (list, tuple, np.ndarray)) and len(tgt_idx) > 0) else None
                        alignment = {
                            'source': src_text,
                            'target': target_text,
                            'source_idx': current_src_idx,
                            'target_idx': current_src_idx
                        }
                        alignments.append(alignment)
                        current_src_idx += 1
                    continue
                alignments.append(alignment)
                current_src_idx += 1
                
            except Exception as align_error:
                print(f"Alignment error details: {align_error}")
                raise HTTPException(
                    status_code=500,
                    detail=f'Error processing alignment: {str(align_error)}'
                )

        while current_src_idx < len(src_sentences):
            alignments.append({
                'source': src_sentences[current_src_idx],
                'target': None,
                'source_idx': current_src_idx,
                'target_idx': current_src_idx
            })
            current_src_idx += 1

        response_data = {
            'status': 'success',
            'alignments': convert_numpy_types(alignments),
            'total_alignments': len(alignments),
            'source_sentences': len(src_sentences),
            'target_sentences': len(aligner.tgt_sents)
        }
        return response_data

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                'error': f'An error occurred: {str(e)}',
                'error_type': type(e).__name__
            }
        )

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8080)
