import sys
import src.data_reader as data_reader
import sqlite3
import src.database as database
from src.preprocess import Preprocess
from src.model import LDA

def test_url(urlpath):
	data = data_reader.get_html_data(urlpath)
	return data

def test_tfidf_vectorizer(filtered_texts):
	vectorized_texts = filtered_texts.vectorize_tfidf()
	return vectorized_texts, filtered_texts.tfidf.get_feature_names_out()

def test_model_fit(vectorized_texts, feature_names):
	build_model = LDA(vectorized_texts, feature_names, n_components=5, max_iter=5, learning_method='online')
	build_model.fit_model()
	topics = build_model.get_topics(5)
	database.save_topics_to_db(topics, "temp.db")
	assignments = build_model.get_topic_assignments()
	database.save_assignments_to_db(assignments, "temp.db")

def test_db(dbname: str, table_name: str):
	con = sqlite3.connect(dbname)
	cur = con.cursor()
	cur.execute("SELECT * FROM {}".format(table_name))
	return cur.fetchall()
	

# Test full pipeline

#Test URL text extraction
url = "https://americanliterature.com/childrens-stories/goldilocks-and-the-three-bears"
data = test_url(url)
print("\n Data extraction check")
print(data[0])


#Test sentence splitter, processing and filtering
preprocessor = Preprocess([data])
preprocessor.lemmatize_and_filter()
print("\n Document split check")
print(preprocessor.sentences[:2])

print("\n Preprocessing")
print(preprocessor.filtered_texts[:2])


#Test sentiment analysis and database
sentiments = preprocessor.get_sentiment()
database.save_documents_to_db(preprocessor.sentences, "temp.db")
database.save_sentiments_to_db(sentiments, "temp.db")
res = test_db("temp.db", "sentiment_analysis")
print("\n Sentiment Analysis")
print(res[0])


#Test vectorizer
v_texts, features = test_tfidf_vectorizer(preprocessor)
print("\n Vectorized shape")
print(v_texts.shape)


#Fit Model and test topics
test_model_fit(v_texts, features)
res = test_db("temp.db", "lda_topics")
print("\n Topics")
print(res[:2])

#check assignments
print("\n Documents assigned to topics")
res = test_db("temp.db", "assigned_topics")
print(res[:2])





#