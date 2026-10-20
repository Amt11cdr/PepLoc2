"""
PeptideLocator2 — Ensemble inference script
Amritansh Tiwari & Raúl Fernández-Díaz — UCD Shields Lab, 2026
Usage: python predict.py --fasta input.fasta --output results.json
"""

import argparse
import json
import os
import re
import gc
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import T5Tokenizer, T5EncoderModel, AutoModel, AutoTokenizer
import esm

REPO_ID = 'Amt11cdr/PeptideLocator2'
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BACKBONES = {
    'prot_t5':   {'dim': 1024, 'mcc': 0.6988},
    'esm2_3b':   {'dim': 2560, 'mcc': 0.6895},
    'esmc_300m': {'dim': 960,  'mcc': 0.6349},
    'esmc_600m': {'dim': 1152, 'mcc': 0.6466},
}

class BiGRUAttnHead(nn.Module):
    def __init__(self, in_features, hidden=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.bigru         = nn.GRU(in_features, hidden, num_layers=2,
                                    batch_first=True, bidirectional=True,
                                    dropout=dropout)
        self.norm1         = nn.LayerNorm(hidden * 2)
        self.attn          = nn.MultiheadAttention(hidden*2, num_heads,
                                                   dropout=dropout, batch_first=True)
        self.drop          = nn.Dropout(dropout)
        self.norm2         = nn.LayerNorm(hidden * 2)
        self.peptide_head  = nn.Linear(hidden * 2, 1)
        self.cleavage_head = nn.Linear(hidden * 2, 1)

    def forward(self, x):
        g, _ = self.bigru(x)
        g     = self.norm1(g)
        a, _ = self.attn(g, g, g)
        x2    = self.norm2(g + self.drop(a))
        return self.peptide_head(x2).squeeze(-1), self.cleavage_head(x2).squeeze(-1)

def parse_fasta(path):
    sequences = {}
    current_id, current_seq = None, []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    sequences[current_id] = ''.join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
    if current_id:
        sequences[current_id] = ''.join(current_seq)
    return sequences

def embed_prott5(sequences):
    print("  Extracting ProtT5-XL embeddings...")
    tokenizer = T5Tokenizer.from_pretrained(
        'Rostlab/prot_t5_xl_uniref50', do_lower_case=False)
    model = T5EncoderModel.from_pretrained(
        'Rostlab/prot_t5_xl_uniref50').to(DEVICE).eval()
    embeddings = {}
    for uid, seq in sequences.items():
        seq_spaced = ' '.join(list(re.sub(r"[UZOB]", "X", seq)))
        ids = tokenizer(seq_spaced, return_tensors='pt').input_ids.to(DEVICE)
        with torch.no_grad():
            out = model(input_ids=ids)
        embeddings[uid] = out.last_hidden_state[0, :len(seq), :].cpu()
    del model; gc.collect(); torch.cuda.empty_cache()
    return embeddings

def embed_esm2_3b(sequences):
    print("  Extracting ESM2-3B embeddings...")
    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    model = model.to(DEVICE).eval()
    batch_converter = alphabet.get_batch_converter()
    embeddings = {}
    for uid, seq in sequences.items():
        _, _, tokens = batch_converter([("p", seq)])
        with torch.no_grad():
            out = model(tokens.to(DEVICE), repr_layers=[36])
        embeddings[uid] = out['representations'][36][0, 1:len(seq)+1, :].cpu()
    del model; gc.collect(); torch.cuda.empty_cache()
    return embeddings

def embed_esmc(sequences, hf_name, label):
    print(f"  Extracting {label} embeddings...")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModel.from_pretrained(hf_name).to(DEVICE).eval()
    embeddings = {}
    for uid, seq in sequences.items():
        ids = tokenizer(seq, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            out = model(**ids)
        embeddings[uid] = out.last_hidden_state[0, 1:len(seq)+1, :].cpu()
    del model; gc.collect(); torch.cuda.empty_cache()
    return embeddings

def get_spans(probs, threshold=0.5, min_len=5):
    binary = (probs > threshold).astype(int)
    spans = []
    in_pep, start = False, 0
    for i, b in enumerate(binary):
        if b == 1 and not in_pep:
            start = i; in_pep = True
        elif b == 0 and in_pep:
            length = i - start
            if length >= min_len:
                spans.append({'start': start+1, 'end': i, 'length': length})
            in_pep = False
    if in_pep:
        length = len(binary) - start
        if length >= min_len:
            spans.append({'start': start+1, 'end': len(binary), 'length': length})
    return spans

def main():
    parser = argparse.ArgumentParser(
        description='PeptideLocator2: Endogenous peptide region prediction')
    parser.add_argument('--fasta',     required=True, help='Input FASTA file')
    parser.add_argument('--output',    required=True, help='Output JSON file')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Prediction threshold (default 0.5)')
    parser.add_argument('--min_len',   type=int, default=5,
                        help='Minimum peptide length in AA (default 5)')
    args = parser.parse_args()

    sequences = parse_fasta(args.fasta)
    print(f"Loaded {len(sequences)} sequences | Device: {DEVICE}")

    all_embeddings = {
        'prot_t5':   embed_prott5(sequences),
        'esm2_3b':   embed_esm2_3b(sequences),
        'esmc_300m': embed_esmc(sequences, 'biohub/ESMC-300M', 'ESMC-300M'),
        'esmc_600m': embed_esmc(sequences, 'biohub/ESMC-600M', 'ESMC-600M'),
    }

    print("\nRunning MCC-weighted ensemble inference...")
    results = {}

    for uid, seq in sequences.items():
        weighted_probs = np.zeros(len(seq))
        total_weight   = 0.0

        for backbone, config in BACKBONES.items():
            emb = all_embeddings[backbone].get(uid)
            if emb is None:
                continue
            weight     = config['mcc']
            fold_probs = []

            for fold in range(5):
                ckpt_path = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=f'checkpoints/best_model_{backbone}_fold{fold}.pt',
                    cache_dir=os.path.expanduser('~/.peptidelocator2/weights')
                )
                model = BiGRUAttnHead(in_features=config['dim']).to(DEVICE)
                model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
                model.eval()
                with torch.no_grad():
                    pep_logits, _ = model(emb.unsqueeze(0).to(DEVICE))
                    fold_probs.append(torch.sigmoid(pep_logits)[0].cpu().numpy())
                del model

            weighted_probs += weight * np.mean(fold_probs, axis=0)
            total_weight   += weight

        final_probs = weighted_probs / total_weight
        spans = get_spans(final_probs, args.threshold, args.min_len)

        results[uid] = {
            'sequence':      seq,
            'length':        len(seq),
            'peptides':      spans,
            'probabilities': final_probs.tolist()
        }
        print(f"  {uid}: {len(spans)} peptide(s) predicted")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")
    print(f"Ensemble: MCC-weighted soft vote (4 backbones x 5 folds = 20 models)")

if __name__ == '__main__':
    main()
