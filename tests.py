import sys
import data_reader
import sqlite3
from preprocess import Preprocess
from model import LDA

def test_csv(filepath):
	data = data_reader.get_csv_data(filepath, ['companyName', 'filedAt', 'Section1', 'Section1A', 'Section7'])
	merged = data_reader.concatenate_columns(data, ['Section1', 'Section1A', 'Section7'])
	year = data['filedAt'].apply(lambda x: x.split('-')[0])
	new_field = [("Year", x) for x in year]
	return merged['merged_text'].values.tolist()[:10], new_field

def test_url(urlpath):
	data = data_reader.get_html_data(urlpath)
	return data

def test_count_vectorizer(filtered_texts):
	vectorized_texts = filtered_texts.vectorize_count()
	return vectorized_texts, filtered_texts.count_vectorizer.get_feature_names_out()

def test_tfidf_vectorizer(filtered_texts):
	vectorized_texts = filtered_texts.vectorize_tfidf()
	return vectorized_texts, filtered_texts.tfidf.get_feature_names_out()

def test_model_fit(vectorized_texts, feature_names):
	build_model = LDA(vectorized_texts, feature_names, n_components=5, max_iter=5, learning_method='online')
	build_model.fit_model()
	topics = build_model.get_topics(5)
	data_reader.save_topics_to_db(topics, "text_analysis.db")
	assignments = build_model.get_topic_assignments()
	data_reader.save_assignments_to_db(assignments, "text_analysis.db")

def test_db(dbname: str, table_name: str):
	con = sqlite3.connect(dbname)
	cur = con.cursor()
	cur.execute("SELECT * FROM {}".format(table_name))
	res = cur.fetchall()
	print(res)

def test_cond_db(dbname: str):
    con = sqlite3.connect(dbname)
    cur = con.cursor()
    #cur.execute("SELECT document, topic, sentiment, fieldvalue FROM documents, lda_topics, assigned_topics, sentiment_analysis, additional_fields WHERE documents.text_id = sentiment_analysis.text_id AND documents.document_id = assigned_topics.document_id AND assigned_topics.topic_id = lda_topics.topic_id AND documents.text_id = additional_fields.text_id AND additional_fields.fieldname LIKE 'YEAR'")
    cur.execute("SELECT assigned_topics.document_id, topic_id, sentiment FROM assigned_topics, sentiment_analysis WHERE assigned_topics.document_id = sentiment_analysis.document_id")
    res = cur.fetchall()
    print(res)
#processed_texts = test_url("https://en.wikipedia.org/wiki/Tere_Bin_(2022_TV_series)")
#data, year = test_csv(sys.argv[1])
#preprocessor = Preprocess(data)
#preprocessor.lemmatize_and_filter()
#sentiments = preprocessor.get_sentiment()
#print(len(preprocessor.filtered_texts), len(sentiments))
#data_reader.save_documents_to_db(preprocessor.sentences, "text_analysis.db")
#data_reader.save_additional_field_to_db(year, "text_analysis.db")
#data_reader.save_sentiments_to_db(sentiments, "text_analysis.db")
#v_texts, features = test_tfidf_vectorizer(preprocessor)
#print(v_texts.shape)
#test_model_fit(v_texts, features)

#test_db("text_analysis.db", "sentiment_analysis")
#test_db("text_analysis.db", "lda_topics")
test_cond_db("text_analysis.db")
#test_db("text_analysis.db", "assigned_topics")

