import data_reader
import sqlite3
from model import LDA
from preprocess import Preprocess
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

class Params(BaseModel):
    path: str
    path_type: str = "html"
    num_topics: int = 5
    max_iterations: Optional[int] = 10
    topics_topk_words: Optional[int] = 10
    db_name: Optional[str] = "text_analysis.db"
 
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello World"}


@app.post("/extract-themes/")
def read_html(params: Params):
    if params.path_type == "html":
        data = [data_reader.get_html_data(params.path)]
    elif params.path_type == "csv":
        data = data_reader.get_csv_data(params.path)

    preprocessor = Preprocess(data)
    preprocessor.lemmatize_and_filter()
    data_reader.save_documents_to_db(preprocessor.sentences, params.db_name)
    
    sentiments = preprocessor.get_sentiment()
    data_reader.save_sentiments_to_db(sentiments, params.db_name)
    
    vectorized_texts = preprocessor.vectorize_tfidf()
    feature_names = preprocessor.tfidf.get_feature_names_out()
    build_model = LDA(vectorized_texts, feature_names, n_components=params.num_topics, max_iter=params.max_iterations, learning_method='online')
    build_model.fit_model()
    
    topics = build_model.get_topics(params.topics_topk_words)
    data_reader.save_topics_to_db(topics, params.db_name)
    
    assignments = build_model.get_topic_assignments()
    data_reader.save_assignments_to_db(assignments, params.db_name)

@app.get("/return-themes/{db_name}")
def return_extracted_themes(db_name: str = "text_analysis.db"):
    con = sqlite3.connect(db_name)
    cur = con.cursor()
    cur.execute("SELECT * FROM lda_topics")
    return cur.fetchall()
 
@app.get("/return-texts/{db_name}")
def return_texts_with_theme_and_sentiment(db_name: str = "text_analysis.db"):
    con = sqlite3.connect(db_name)
    cur = con.cursor()
    cur.execute("SELECT document, topic, sentiment FROM documents, assigned_topics, lda_topics, sentiment_analysis WHERE documents.document_id = assigned_topics.document_id AND assigned_topics.topic_id = lda_topics.topic_id AND sentiment_analysis.document_id = documents.document_id")
    return cur.fetchall()

@app.get("/return-documents/{db_name}")
def return_saved_documents(db_name: str = "text_analysis.db"):
    con = sqlite3.connect(db_name)
    cur = con.cursor()
    cur.execute("SELECT * FROM documents")
    return cur.fetchall()
  
    
