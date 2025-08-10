from sentence_transformers import SentenceTransformer

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

embeddings=model.encode(corpus)

from sklearn.cluster import KMeans

num_clusters=3
Kmeans=KMeans(n_clusters=num_clusters,random_state=42)
Kmeans.fit(embeddings)

labels=Kmeans.labels_

#TF-IDF(Topic-Modeling)

from sklearn.feature_extraction.text import TfidfVectorizer

cluster_texts = {i: [] for i in range(num_clusters)}
for i, label in enumerate(labels):
    cluster_texts[label].append(corpus[i])

# Keyword extraction
for cluster_id, texts in cluster_texts.items():
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5)
    X = vectorizer.fit_transform(texts)
    keywords = vectorizer.get_feature_names_out()
    print(f"Topic {cluster_id}: {keywords}")