# PepLoc2 — PeptideLocator2

**Endogenous peptide region prediction from protein sequences using frozen PLM embeddings.**

Amritansh Tiwari & Raúl Fernández-Díaz — UCD Shields Lab, University College Dublin

## Overview

PeptideLocator2 predicts endogenous bioactive peptide regions directly from protein sequences. It uses a MCC-weighted ensemble of 20 models (4 PLM backbones × 5 folds) trained on the DeepPeptide dataset with GraphPart cross-validation splits.

## Backbones

- ProtT5-XL (Elnaggar et al., 2021)
- ESM2-3B (Lin et al., 2023)
- ESMC-300M (Hayes et al., 2024)
- ESMC-600M (Hayes et al., 2024)

## Usage

\`\`\`bash
git clone https://github.com/Amt11cdr/PepLoc2
cd PepLoc2
pip install -r requirements.txt
python predict.py --fasta input.fasta --output results.json
\`\`\`

## Arguments

| Argument | Default | Description |
|---|---|---|
| \`--fasta\` | required | Input FASTA file |
| \`--output\` | required | Output JSON file |
| \`--threshold\` | 0.5 | Prediction probability threshold |
| \`--min_len\` | 5 | Minimum peptide length (AA) |

## Model weights

Weights hosted on HuggingFace: [Amt11cdr/PeptideLocator2](https://huggingface.co/Amt11cdr/PeptideLocator2)

Downloaded automatically on first run.

## License

Apache 2.0
