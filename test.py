import torch
from pathlib import Path
d = Path("data/processed/embeddings")
for name in ["train_doc_embs.pt","validation_doc_embs.pt","test_doc_embs.pt"]:
    x = torch.load(d / name, map_location="cpu")
    print(name, x.shape)
EOF