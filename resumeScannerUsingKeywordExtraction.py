#Necessary libraries
import pandas as pd     #Data manipulation and create data frames
from sklearn.feature_extraction.text import TfidfVectorizer     #Keywords extraction
from sklearn.metrics.pairwise import cosine_similarity      #Data comparison

#Required data
resumeData = {
    'resumeId': [1, 2, 3],
    'resumeText': [
        "Detail-oriented and resourceful Full Stack Developer with 10 years of experience in designing, developing, and deploying scalable web applications. Skilled in modern front-end frameworks, robust back-end systems, and cloud-based solutions. Adept at collaborating with cross-functional teams and delivering high-quality software under tight deadlines.",
        "Analytical and detail-oriented Data Scientist with X years of experience leveraging statistical modeling, machine learning, and data visualization to drive business insights and decision-making. Skilled in handling large datasets, building predictive models, and deploying scalable solutions across diverse domains. Proficient in Python, R, SQL, and modern data science frameworks, with a strong background in cloud platforms and big data technologies. Adept at collaborating with cross-functional teams to translate complex data into actionable strategies, optimize processes, and deliver measurable impact. Passionate about continuous learning and applying cutting-edge techniques to solve real-world problems.",
        "Highly skilled Database Engineer with 8 years of experience in designing, implementing, and optimizing relational and non-relational database systems. Proficient in SQL, PL/SQL, and modern database technologies such as MySQL, PostgreSQL, Oracle, and MongoDB. Adept at developing efficient data models, ensuring high availability, and implementing robust backup and recovery strategies. Experienced in performance tuning, query optimization, and managing large-scale datasets to support mission-critical applications. Strong background in collaborating with developers, analysts, and system administrators to deliver secure, scalable, and reliable database solutions. Passionate about leveraging automation, cloud platforms, and DevOps practices to streamline database operations and improve efficiency"

    ]
}

jobDescription = "A Data Scientist is responsible for collecting, cleaning, and analyzing large datasets to uncover patterns, trends, and insights. They design and implement machine learning models, statistical analyses, and data-driven solutions that help organizations improve operations, products, and strategies."

#Convert data into data frames
dataFrames = pd.DataFrame(resumeData)
print("Below are the resumes:\n", dataFrames)

#Combine resumes and job description for TF-IDF Vectorization
documents = dataFrames['resumeText'].tolist()
documents.append(jobDescription)

#Intitialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidfMatrix = vectorizer.fit_transform(documents)

#Calculate similarity scores between resumes and job description
similarityScores = cosine_similarity(tfidfMatrix[-1], tfidfMatrix[:-1]).flatten()

#Display similarity score for each resume
dataFrames['similarityScores'] = similarityScores
print("Resume similarity scores\n", dataFrames[['resumeId', 'similarityScores']])

#Identiy resumes that match job requirements
threshold = 0.2
matchedResume = dataFrames[dataFrames['similarityScores'] >= threshold]
print("Matched resumes with requirements:\n", matchedResume[['resumeId', 'similarityScores']])
