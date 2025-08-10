
import torch

from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Corpus with example documents
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

# Use "convert_to_tensor=True" to keep the tensors on GPU (if available)
corpus_embeddings = embedder.encode_document(corpus)

# Query sentences:
queries = [
    "About labs?",
    "About  classrooms?",
    "I feel we are not allowed to express our doubts freely.",
    "About Mess and food and canteen"
]


# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = min(5, len(corpus))
for query in queries:
    query_embedding = embedder.encode_query(query)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    similarity_scores = embedder.similarity(query_embedding, corpus_embeddings)[0]
    scores, indices = torch.topk(similarity_scores, k=top_k)

    print("\nQuery:", query)
    print("Top 5 most similar sentences in corpus:")

    for score, idx in zip(scores, indices):
        print(f"(Score: {score:.4f})", corpus[idx])

    """
    # Alternatively, we can also use util.semantic_search to perform cosine similarty + topk
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)
    hits = hits[0]      #Get the hits for the first query
    for hit in hits:
        print(corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))
    """