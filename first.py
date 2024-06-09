import joblib
import numpy as np
import pandas as pd
import operator
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin


from sklearn.base import BaseEstimator, TransformerMixin

stop_words = set(stopwords.words('english'))

class JobRecommender:
    def __init__(self):
        self.skills = []
        self.jobs = []
        self.similarity_matrix = None

    def delete_spec_chars(self, input): 
        regex = r'[^a-zA-Z0-9\s]'
        output = re.sub(regex,'',input)
        return output

    def union(self, job1, job2):
        return list(set(job1) | set(job2))

    def intersection(self, job1, job2):
        return list(set(job1) & set(job2))

    def calculate_similarity(self):
        dict_val = {}
        for index, obj in enumerate(self.skills):
            inner_list = []
            for next_index, next_obj in enumerate(self.skills):
                if index == next_index:
                    inner_list.append(1.0)
                else:
                    union_result = self.union(obj, next_obj)
                    intersection_result = self.intersection(obj, next_obj)
                    inner_list.append(len(intersection_result) / len(union_result))
            dict_val[self.jobs[index]] = inner_list
        return dict_val

    def fit(self, data):
        # Preprocessing data
        self.jobs = data['job_title'].unique()
        for j in self.jobs:
            d = []
            for i, row in data.iterrows():
                if row['job_title'] == j:
                    d.append(row['skills'])
            doc = self.delete_spec_chars(str(d))
            doc = re.sub(r'\d+', '', doc)
            doc = re.sub(r'skills', '', doc)
            tokens = word_tokenize(doc)
            tokens = [word.lower() for word in tokens]
            tokens = list(set(tokens))
            self.skills.append(tokens)
        self.similarity_matrix = self.calculate_similarity()

    def calculate_query_similarity(self, skills_query):
        dict = {}
        for index, obj in enumerate(self.skills):
            union_result = self.union(skills_query, obj)
            intersection_result = self.intersection(skills_query, obj)
            dict[self.jobs[index]] = len(intersection_result) / len(union_result)
        return dict

    def recommend(self, skills_input):
        skills_preprocessed = self.delete_spec_chars(skills_input)
        tokens = word_tokenize(skills_preprocessed)
        tokens = [word.lower() for word in tokens]
        skills_query = [word for word in tokens if word not in stop_words and len(word) > 1]
        res = self.calculate_query_similarity(skills_query)
        sorted_d = dict(sorted(res.items(), key=operator.itemgetter(1), reverse=True))
        return sorted_d



# Load data
data = pd.read_csv('jobs.csv')
data = data.drop(columns=['status','city','organization_id','description'])
data = data.dropna()

# Train the recommender
recommender = JobRecommender()
recommender.fit(data)

# Save the model
joblib.dump(recommender, 'job_recommender_model.pkl')


# Load the model
recommender = joblib.load('job_recommender_model.pkl')

# Function to get user input and recommend jobs
def get_recommendations():
    skills_input = input("Please enter your skills: ")
    recommendations = recommender.recommend(skills_input)
    for job, score in recommendations.items():
        print(f"Job Title: {job}, Similarity Score: {score}")

# Get recommendations based on user input
get_recommendations()

