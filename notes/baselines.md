\## LegalBERT cosine baseline (zero-shot)



\- Embeddings: precomputed LegalBERT CLS embeddings for validation/test documents and label descriptions.

\- Similarity: cosine(doc\_emb, label\_emb).

\- Threshold: single global cosine threshold tuned on the validation set.



Validation (n=5000):

\- Macro-F1: 0.0877

\- Micro-F1: 0.0935



Test (n=5000):

\- Macro-F1: 0.0906

\- Micro-F1: 0.0967



Config:

\- Code: scripts/run\_cosine\_baseline.py

\- Log: outputs/logs/baseline\_legalbert\_cosine.json (generated on Ampere)

