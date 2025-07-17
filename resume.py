import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import nltk
import pandas as pd
import re
tfidf=TfidfVectorizer(stop_words='english')
import warnings
warnings.filterwarnings("ignore")

nltk.download('stopwords')
nltk.download('punkt')




#load a model

clf=pickle.load(open('clf.pkl','rb'))
tfidf=pickle.load(open('tfidf.pkl','rb'))


def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('@\S+', '', resumeText)  # remove @
    resumeText = re.sub('#\S+', '', resumeText)  # remove the hashtags
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace

    return resumeText

#web App

def main():
    st.title('Resume Screen App')
    uploaded_files=st.file_uploader('Resume Upload',type=['txt','pdf'])

    if uploaded_files is not None:
        try:
            resume_bytes=uploaded_files.read()
            resumeText=resume_bytes.decode('utf-8')

        except UnicodeDecodeError:
            resumeText=resume_bytes.decode('latin-1')

        cleaned_resume=cleanResume(resumeText)
        cleaned_resume=tfidf.transform([cleaned_resume])
        prediction_id=clf.predict(cleaned_resume)[0]
        st.write(prediction_id)
        category_mapping = {


            'Data Science': 6,
            'HR': 12,
            'Advocate': 0,
            'Arts': 1, 
            'Web Designing': 24,
            'Mechanical Engineer': 16,
            'Sales': 22,
            'Health and fitness': 14,
            'Civil Engineer': 5,
            'Java Developer': 15,
            'Business Analyst': 4,
            'SAP Developer': 21,
            'Automation Testing': 2,
            'Electrical Engineering': 11,
            'Operations Manager': 18,
            'Python Developer': 20,
            'DevOps Engineer': 8,
            'Network Security Engineer': 17,
            'PMO': 19,
            'Database': 7,
            'Hadoop': 13,
            'ETL Developer': 10,
            'DotNet Developer': 9,
            'Blockchain': 3,
            'Testing': 23
        }
        

        label_to_category = {v: k for k, v in category_mapping.items()}
        category_name=label_to_category.get(prediction_id, "unknown")

        st.write("Predicted Category:",category_name)







#python main

if __name__=="__main__":
    main()
