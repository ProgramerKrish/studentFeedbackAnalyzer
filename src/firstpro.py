from sentence_transformers import SentenceTransformer
import numpy as np
import torch

model=SentenceTransformer("output/sbert-finetuned-feedback")

corpus = [
    # Academics - Positive
    "The professor explains concepts very clearly with real-life examples.",
    "Some teachers use interactive teaching methods that keep students engaged.",
    "I really appreciate the detailed feedback on our assignments.",
    "The faculty is friendly, supportive, and approachable for doubts.",
    "The guest lectures are insightful and cover industry-relevant topics.",
    "Practical sessions in the lab are well-organized and hands-on.",
    "Workshops conducted by external experts are very useful for learning.",
    "The library offers a wide range of updated reference books and journals.",
    "Our department organizes regular coding competitions and hackathons.",
    "The academic curriculum includes both theoretical and practical learning.",
    
    # Academics - Negative
    "The syllabus is outdated and needs to be revised to match industry trends.",
    "Some professors rush through topics without ensuring students understand.",
    "The exam pattern is confusing and not aligned with what is taught.",
    "We don't have enough project guidance from the faculty.",
    "Assignments are too frequent, stressful, and time-consuming.",
    "The teaching style in most subjects is overly theoretical.",
    "Not enough emphasis is placed on developing problem-solving skills.",
    "Some professors don’t encourage open discussions in class.",
    "There is a lack of interdisciplinary learning opportunities.",
    "The grading system is not transparent and feels inconsistent.",
    
    # Facilities - Positive
    "The campus is clean, green, and well-maintained.",
    "The classrooms are equipped with projectors and comfortable seating.",
    "The computer labs have high-speed internet and good software availability.",
    "The library has a quiet and peaceful study environment.",
    "The hostel rooms are spacious and well-ventilated.",
    "The sports complex is modern and well-equipped.",
    "The cafeteria offers a variety of food options at reasonable prices.",
    "The college provides reliable transport facilities for students.",
    "The Wi-Fi connection in academic buildings is fast and stable.",
    "The auditorium is large and suitable for big events.",
    
    # Facilities - Negative
    "The classrooms are too small and often congested.",
    "Many classrooms lack proper ventilation and lighting.",
    "The food in the canteen is not hygienic and lacks variety.",
    "The sports facilities are poorly maintained and outdated.",
    "The internet connection in the hostel is very slow.",
    "Lab equipment often breaks down and is not repaired quickly.",
    "Some classrooms do not have working projectors or speakers.",
    "The hostel bathrooms are not cleaned regularly.",
    "The library has limited seating during exam season.",
    "The drinking water facilities on campus are not properly maintained.",
    
    # Student Life - Positive
    "The cultural fest is vibrant and well-organized every year.",
    "The student clubs are active and encourage participation.",
    "There are plenty of extracurricular activities for skill development.",
    "The music and drama clubs put on impressive performances.",
    "The debate club helps improve public speaking skills.",
    "The photography club captures campus life beautifully.",
    "There are regular inter-college sports competitions.",
    "The student council is approachable and addresses concerns quickly.",
    "Peer learning sessions help students share knowledge effectively.",
    "Volunteer programs encourage community service among students.",
    
    # Student Life - Negative
    "There are not enough recreational spaces for students to relax.",
    "The number of student events has reduced compared to previous years.",
    "Clubs often face budget constraints limiting their activities.",
    "Some student activities are not well-promoted, leading to low turnout.",
    "The timing of events often clashes with academic schedules.",
    "There is no dedicated mental health support team on campus.",
    "Students have limited access to counseling services.",
    "The hostel curfew is too strict and limits evening activities.",
    "Some extracurricular events feel repetitive and lack creativity.",
    "Student grievances sometimes go unanswered by management.",
    
    # Administration - Positive
    "The admission process was smooth and well-organized.",
    "Fee payment and other formalities are handled efficiently online.",
    "The placement cell actively connects students with recruiters.",
    "The administration supports students in participating in external competitions.",
    "The examination department announces schedules well in advance.",
    "The academic calendar is usually followed without major delays.",
    "The management is open to feedback from students.",
    "Alumni network events help with career guidance.",
    "Industry partnerships bring internship opportunities.",
    "Faculty training programs improve teaching quality.",
    
    # Administration - Negative
    "The placement opportunities have reduced significantly.",
    "Some administrative staff are unhelpful and slow in processing requests.",
    "It takes too long to get official documents like transcripts.",
    "There is a lack of transparency in fee hikes.",
    "Student complaints are sometimes ignored by higher authorities.",
    "The grievance redressal system is not efficient.",
    "Important notices are often shared at the last minute.",
    "The hostel allocation process is poorly managed.",
    "Scholarship application processing is slow.",
    "Event approvals take too long, delaying student initiatives."
]

embedding_corpus=model.encode_document(corpus)
embedding_corpus=np.array(embedding_corpus,dtype=np.float64)

from sklearn.cluster import KMeans

num_cluster=5
kmeans=KMeans(n_clusters=num_cluster,random_state=42)
labels=kmeans.fit_predict(embedding_corpus)


def loop():

    while True:

        query=input("PROMPT:")

        if query:
            embedding_input=model.encode_query(query)
            embedding_input=np.array(embedding_input,dtype=np.float64).reshape(1, -1)
        else :
            return -1
        
        cluster_id=kmeans.predict(embedding_input)[0]
        print(cluster_id)
        
        cluster_idx=[i for i,label in enumerate(labels) if label==cluster_id]
        """ cluster_embeddings=[embedding_corpus[i] for i in cluster_idx] """
        cluster_embeddings=embedding_corpus[cluster_idx]
        
        

        similarity_scores=model.similarity(embedding_input,cluster_embeddings)[0]
        top_k=min(3,len(cluster_idx))

        scores,indices=torch.topk(similarity_scores,k=top_k)

        print("\nQuery:",query)

        for score,idx in zip(scores,indices):
            real_idx=cluster_idx[idx]
            print(f"(score:{score:.4f})",corpus[real_idx])
            
loop()
