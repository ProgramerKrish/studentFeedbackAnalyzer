""" from sentence_transformers import SentenceTransformer

import torch

model=SentenceTransformer("all-MiniLM-L6-v2")

corpus = [
    "The professor explains concepts very clearly.",
    "The classrooms are too small and congested.",
    "We need more practical sessions in the lab.",
    "The food in the canteen is not hygienic.",
    "Some professors don’t allow us to ask doubts freely.",
    "The syllabus is outdated and needs revision.",
    "The internet connection in the hostel is very slow.",
    "I really enjoy the coding club activities.",
    "The campus is clean and well-maintained.",
    "We are getting fewer placement opportunities.",
    "The assignments are too frequent and stressful.",
    "The faculty is friendly and supportive.",
    "There is no proper ventilation in the library.",
    "The exam pattern is very confusing.",
    "Workshops and guest lectures are very useful.",
    "The sports facilities are not maintained properly.",
    "We don't have enough project guidance from faculty.",
    "The lab equipment is often not working.",
    "Many classrooms lack proper lighting.",
    "The teaching style is mostly theoretical."
]
embedding_corpus=model.encode_document(corpus)

def loop():

    while True:

        query=input("PROMPT:")

        if query:
            embedding_input=model.encode_query(query)
        else :
            return -1

        top_k=min(3,len(corpus))

        similarity_scores=model.similarity(embedding_input,embedding_corpus)[0]
        scores,indices=torch.topk(similarity_scores,k=top_k)

        print("\nQuery:",query)

        for score,idx in zip(scores,indices):
            print(f"(score:{score:.4f})",corpus[idx])
            
loop()

"""
from sklearn.cluster import KMeans
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
model=SentenceTransformer("all-MiniLM-L6-v2")
# Step 1: Encode corpus
corpus = [
    "The professor explains concepts very clearly.",
    "The classrooms are too small and congested.",
    "We need more practical sessions in the lab.",
    "The food in the canteen is not hygienic.",
    "Some professors don’t allow us to ask doubts freely.",
    "The syllabus is outdated and needs revision.",
    "The internet connection in the hostel is very slow.",
    "I really enjoy the coding club activities.",
    "The campus is clean and well-maintained.",
    "We are getting fewer placement opportunities.",
    "The assignments are too frequent and stressful.",
    "The faculty is friendly and supportive.",
    "There is no proper ventilation in the library.",
    "The exam pattern is very confusing.",
    "Workshops and guest lectures are very useful.",
    "The sports facilities are not maintained properly.",
    "We don't have enough project guidance from faculty.",
    "The lab equipment is often not working.",
    "Many classrooms lack proper lighting.",
    "The teaching style is mostly theoretical."
]


embedding_corpus = model.encode_document(corpus)

# Step 2: Cluster into topics
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(embedding_corpus)

# Step 3: Query loop
def loop():
    while True:
        query = input("PROMPT: ").strip()
        if not query:
            break
        
        embedding_input = model.encode_query(query)
        
        # Find closest cluster
        cluster_id = kmeans.predict([embedding_input])[0]
        
        # Filter corpus by that cluster
        cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
        cluster_embeddings = [embedding_corpus[i] for i in cluster_indices]
        
        # Compute similarity within that cluster
        similarity_scores = model.similarity(embedding_input, cluster_embeddings)[0]
        top_k = min(3, len(cluster_indices))
        scores, indices = torch.topk(similarity_scores, k=top_k)
        
        print("\nQuery:", query)
        for score, idx in zip(scores, indices):
            real_idx = cluster_indices[idx]
            print(f"(score:{score:.4f}) {corpus[real_idx]}")

loop()
