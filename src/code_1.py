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
num_cluster=5
kmeans=KMeans(n_clusters=num_cluster,random_state=42)
kmeans.fit(embeddings)

labels=kmeans.labels_

print(labels)

from sklearn.feature_extraction.text import TfidfVectorizer

cluster_text={i:[] for i in range (num_cluster)}
for i,label in enumerate(labels):
    cluster_text[label].append(corpus[i])
    
for cluster_id,text in cluster_text.items():
    Vectorizer=TfidfVectorizer(stop_words="english",max_features=5,ngram_range=(1,2))
    x=Vectorizer.fit_transform(text)
    keywords=Vectorizer.get_feature_names_out()
    print(f"topic {cluster_id}:{keywords}")