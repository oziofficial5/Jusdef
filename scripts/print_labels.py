import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess.data_loader import get_label_names
from src.preprocess.label_loader import load_label_texts

names = get_label_names()
texts = load_label_texts(names)

for i, name in enumerate(names):
    print(f"{i:3d}: {name:>8s} -> {texts[i]}")