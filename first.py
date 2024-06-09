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

class JobRecommender(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.jobs = None
        self.skills = None
        self.tanimoto_sim_dict = None

    def fit(self, X, y=None):
        self.jobs = X['job_title'].unique()
        newData = X[['user_id', 'job_title', 'skills']]
        
        self.skills = []
        for j in self.jobs:
            d = []
            for i, row in newData.iterrows():
                if(row['job_title'] == j):
                    d.append(row['skills'])
            
            doc = self._delete_spec_chars(str(d))
            doc = re.sub(r'\d+', '', doc)
            doc = re.sub(r'skills', '', doc)
            
            tokens = word_tokenize(doc)
            tokens = [word.lower() for word in tokens]
            tokens = list(set(tokens))
            
            self.skills.append(tokens)
        
        self.tanimoto_sim_dict = self._calculate_tanimoto_similarity()
        return self
    
    def _delete_spec_chars(self, input):
        regex = r'[^a-zA-Z0-9\s]'
        output = re.sub(regex, '', input)
        return output
    
    def _union(self, job1, job2):
        return list(set(job1) | set(job2))
    
    def _intersection(self, job1, job2):
        return list(set(job1) & set(job2))
    
    def _calculate_tanimoto_similarity(self):
        dict_tanimoto_val = {}
        for index, obj in enumerate(self.skills):
            inner_list = []
            for next_index, next_obj in enumerate(self.skills):
                if index == next_index:
                    inner_list.append(1.0)
                else:
                    union_result = self._union(obj, next_obj)
                    intersection_result = self._intersection(obj, next_obj)
                    inner_list.append(len(intersection_result) / len(union_result))
            dict_tanimoto_val[self.jobs[index]] = inner_list
        return dict_tanimoto_val

    def _calculate_query_similarity(self, skills_query):
        tanimoto_dict = {}
        for index, obj in enumerate(self.skills):
            union_result = self._union(skills_query, obj)
            intersection_result = self._intersection(skills_query, obj)
            tanimoto_dict[self.jobs[index]] = (len(intersection_result) / len(union_result))
        return tanimoto_dict
    
    def recommend(self, skills_input):
        skills_preprocessed = self._delete_spec_chars(skills_input)
        tokens = word_tokenize(skills_preprocessed)
        tokens = [word.lower() for word in tokens]
        skills_query = [word for word in tokens if word not in stop_words and len(word) > 1]
        
        tanimoto_val = self._calculate_query_similarity(skills_query)
        sorted_d = dict(sorted(tanimoto_val.items(), key=operator.itemgetter(1), reverse=True))
        
        return sorted_d


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
