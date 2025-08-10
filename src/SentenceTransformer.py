""" import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU") """





from sentence_transformers import SentenceTransformer
import pandas
import numpy

model=SentenceTransformer("all-MiniLM-L6-v2")


sentence1=[
    "The new movie is awesome",
    "The cat sits outside",
    "A man is playing guitar",
]

sentence2= [
    "The dog plays in the garden",
    "The new movie is so great",
    "A woman watches TV",
]

embeddings1=model.encode(sentence1)
embeddings2=model.encode(sentence2)

similarities=model.similarity(embeddings1,embeddings2)

for i,sentence in enumerate(sentence1):
    print(sentence)
    for j,_ in enumerate(sentence2):
        print(f" -{_ : <30}:{similarities[i][j]}")

 

