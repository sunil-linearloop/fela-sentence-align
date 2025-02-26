from flask import Flask, request, jsonify
from bertalign import Bertalign
from typing import Dict, Any
import numpy as np
import re

app = Flask(__name__)

def convert_numpy_types(obj):
    """Convert numpy types to native Python types."""
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
    """Split text into sentences more accurately."""
    # Split on period followed by space and uppercase letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    return [sent.strip() for sent in sentences if sent.strip()]

@app.route('/align', methods=['POST'])
def align_texts():
    try:
        # Get JSON data from request with explicit error handling
        try:
            data = request.get_json(force=True)  # force=True will handle different content types
        except Exception as json_error:
            return jsonify({
                'error': 'Invalid JSON format in request body',
                'details': str(json_error)
            }), 400
        
        # Validate input with more specific error messages
        if not isinstance(data, dict):
            return jsonify({
                'error': 'Request body must be a JSON object'
            }), 400
            
        if not data:
            return jsonify({
                'error': 'Request body cannot be empty'
            }), 400
            
        if 'src' not in data or 'tgt' not in data:
            missing_fields = []
            if 'src' not in data:
                missing_fields.append('src')
            if 'tgt' not in data:
                missing_fields.append('tgt')
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
            
        # Ensure src and tgt are strings
        if not isinstance(data['src'], str) or not isinstance(data['tgt'], str):
            return jsonify({
                'error': 'Both "src" and "tgt" must be strings'
            }), 400
            
        src_text = data['src'].strip()
        tgt_text = data['tgt'].strip()
        
        # Pre-split the source text into sentences
        src_sentences = split_into_sentences(src_text)
        
        # Create aligner with pre-split source sentences
        aligner = Bertalign("\n".join(src_sentences), tgt_text)
        aligner.align_sents()
        
        # # Debug print to see raw Bertalign output
        # print("Raw Bertalign alignments:", aligner.result)
        # print("Source sentences:", src_sentences)
        # print("Target sentences:", aligner.tgt_sents)

        # Get alignment results with additional error handling
        alignments = []
        current_src_idx = 0
        
        for src_idx, tgt_idx in aligner.result:
            try:
                # Fill in any gaps in source indices
                while current_src_idx < (src_idx if isinstance(src_idx, (np.integer, int)) else src_idx[0]):
                    alignments.append({
                        'source': src_sentences[current_src_idx],
                        'target': None,
                        'source_idx': current_src_idx,
                        'target_idx': current_src_idx
                    })
                    current_src_idx += 1
                
                # Handle single sentence alignment
                if isinstance(src_idx, (np.integer, int)) and isinstance(tgt_idx, (np.integer, int)):
                    alignment = {
                        'source': src_sentences[src_idx],
                        'target': aligner.tgt_sents[tgt_idx],
                        'source_idx': current_src_idx,
                        'target_idx': current_src_idx
                    }
                    current_src_idx += 1
                # Handle many-to-many sentence alignment
                else:
                    src_text_list = [src_sentences[i] for i in (src_idx if isinstance(src_idx, (list, tuple, np.ndarray)) else [src_idx])]
                    # Split each source sentence into its own alignment
                    for i, src_text in enumerate(src_text_list):
                        # Only assign target text to the first sentence in the group
                        target_text = aligner.tgt_sents[tgt_idx[0]] if i == 0 and isinstance(tgt_idx, (list, tuple, np.ndarray)) and len(tgt_idx) > 0 else None
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
                
            except Exception as align_error:
                print(f"Alignment error details: {align_error}")  # Debug print
                return jsonify({
                    'error': f'Error processing alignment: {str(align_error)}'
                }), 500

        # Fill in any remaining source indices at the end
        while current_src_idx < len(src_sentences):
            alignments.append({
                'source': src_sentences[current_src_idx],
                'target': None,
                'source_idx': current_src_idx,
                'target_idx': current_src_idx
            })
            current_src_idx += 1

        # Convert response data before jsonify
        response_data = {
            'status': 'success',
            'alignments': convert_numpy_types(alignments),
            'total_alignments': len(alignments),
            'source_sentences': len(src_sentences),
            'target_sentences': len(aligner.tgt_sents)
        }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({
            'error': f'An error occurred: {str(e)}',
            'error_type': type(e).__name__
        }), 500

if __name__ == '__main__':
    app.run(debug=True)