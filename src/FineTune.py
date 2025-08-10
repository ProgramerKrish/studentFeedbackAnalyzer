from sentence_transformers import SentenceTransformer , losses ,InputExample
import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import util
import random


# Positive feedback sentences
positive_sentences = [
    "The professor explains concepts clearly.",
    "The teacher is good at explaining ideas.",
    "The classrooms are spacious and comfortable.",
    "The canteen food is tasty and fresh.",
    "We have plenty of hands-on lab sessions.",
    "The library is well stocked with books.",
    "The sports facilities are excellent.",
    "Wi-Fi is fast and reliable.",
    "The syllabus is well structured.",
    "Assignments are very helpful."
]

# Negative feedback sentences
negative_sentences = [
    "The professor is unclear and confusing.",
    "The classrooms are too small and crowded.",
    "The canteen food is bad and stale.",
    "We have very few lab sessions.",
    "The library has outdated books.",
    "Sports facilities are poorly maintained.",
    "Wi-Fi is slow and unreliable.",
    "The syllabus is disorganized.",
    "Assignments are useless.",
    "Lectures are boring and unengaging."
]

def generate_pairs(n_pairs=1000):
    data = []
    for _ in range(n_pairs):
        if random.random() > 0.5:
            # Positive pair (similar, label=1.0)
            s1 = random.choice(positive_sentences)
            s2 = random.choice(positive_sentences)
            label = 1.0
        else:
            # Negative pair (dissimilar, label=0.0)
            s1 = random.choice(positive_sentences)
            s2 = random.choice(negative_sentences)
            label = 0.0
        
        data.append([s1, s2, label])
    return pd.DataFrame(data, columns=["sentence1", "sentence2", "label"])

# Generate dataset
df = generate_pairs(1000)

# Save CSV
df.to_csv("feedback_pairs_1k.csv", index=False)
print("CSV saved as feedback_pairs_1k.csv with", len(df), "rows")




df=pd.read_csv("feedback_pairs_1k.csv")

train_examples = [
    InputExample(texts=[row['sentence1'], row['sentence2']], label=float(row['label']))
    for _, row in df.iterrows()
]

model=SentenceTransformer("all-MiniLM-L6-v2")
emb1=model.encode("The professor explains well.", convert_to_tensor=True)
emb2 = model.encode("The teacher is good at explaining concepts.", convert_to_tensor=True)
similarity=util.pytorch_cos_sim(emb1,emb2)
print("Before :",similarity.item())

train_loss=losses.CosineSimilarityLoss(model)


train_dataloader=DataLoader(train_examples,shuffle=True,batch_size=16)

model.fit(
    train_objectives=[(train_dataloader,train_loss)],
    epochs=1,
    warmup_steps=int(len(train_dataloader)*0.1), # 10% warmup
    output_path="output/sbert-finetuned-feedback"
    
)



finetuned_model=SentenceTransformer("output/sbert-finetuned-feedback")

emb1=finetuned_model.encode("The professor explains well.", convert_to_tensor=True)
emb2 = finetuned_model.encode("The teacher is good at explaining concepts.", convert_to_tensor=True)

similarity=util.pytorch_cos_sim(emb1,emb2)
print("similarity:",similarity.item())

model.save("output/sbert-finetuned-feedback")