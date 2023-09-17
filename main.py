import src.data_reader as data_reader
import src.database as database
import sqlite3
import pandas as pd
from src.model import LDA
from src.preprocess import Preprocess
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


description = """
## Text Analysis API can:

* **Scrape data from HTML and infer topics**
* **Save inferred topics, sentiments and documents to database**
* **Retrieve**

### **/extract-themes**
* **Minimally expects URL path and can also take other parameters for topic model**
* **Automatically runs topic modeling and sentiment analysis**
* **Saves all data in sqlite3 database**
* **path: URL link**
* **path_type: html**
* **num_topics: Number of topics to fit for, default is 5**
* **max_iterations: Number of iterations for the model, default=10**
* **topics_topk_words: Number of representative words to choose per topic, default=10**
* **db_name: Specify database name for saving, default is `text_analysis.db`**


### **/return-themes**
* **Returns extracted themes**
* **Specify database name if passed during extraction or use `text_analysis.db`**

### **/return-texts**
* **Returns documents, their sentiment and assigned themes**
* **Specify database name if passed during extraction or use `text_analysis.db`**

### **/return-documents**
* **Returns list of extracted texts before processing**
* **Specify database name if passed during extraction or use `text_analysis.db`**
"""

 
app = FastAPI(title="Text Analysis", description=description)

@app.get("/")
def root():
    return {"message": "Welcome to Text Analysis."}


@app.post("/extract-themes/")
def read_html(params: Params):
    if params.path_type == "html":
        data = [data_reader.get_html_data(params.path)]
    elif params.path_type == "csv":
        data = data_reader.get_csv_data(params.path)

    preprocessor = Preprocess(data)
    preprocessor.lemmatize_and_filter()
    database.save_documents_to_db(preprocessor.sentences, params.db_name)
    
    sentiments = preprocessor.get_sentiment()
    database.save_sentiments_to_db(sentiments, params.db_name)
    
    vectorized_texts = preprocessor.vectorize_tfidf()
    feature_names = preprocessor.tfidf.get_feature_names_out()
    build_model = LDA(vectorized_texts, feature_names, n_components=params.num_topics, max_iter=params.max_iterations, learning_method='online')
    build_model.fit_model()
    
    topics = build_model.get_topics(params.topics_topk_words)
    database.save_topics_to_db(topics, params.db_name)
    
    assignments = build_model.get_topic_assignments()
    database.save_assignments_to_db(assignments, params.db_name)

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
  
    
