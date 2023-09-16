import sys
import requests
import sqlite3
import pandas as pd
from bs4 import BeautifulSoup

def get_csv_data(filepath: str, columns: list = None) -> pd.DataFrame:
	rows = pd.read_csv(filepath)
	if columns:
		return rows[columns]
	else:
		return rows

def concatenate_columns(rows: list, merge_columns: list) -> pd.DataFrame:
	rows["merged_text"] = rows[merge_columns].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
	return rows


def get_html_data(urlpath: str):
	response = requests.get(urlpath)
	soup = BeautifulSoup(response.text, features='html.parser')
	for script in soup(["script", "style"]):
		script.extract()
	return soup.get_text()

def save_documents_to_db(documents: list, dbname: str):
	con = sqlite3.connect(dbname)
	cur = con.cursor()
	cur.execute("CREATE TABLE IF NOT EXISTS documents(document_id, text_id, document)")
	for i, document in enumerate(documents):
		print(i, document[0], document[1])
		cur.execute("INSERT INTO documents VALUES(?, ?, ?)", (i, document[0], document[1]))
	con.commit()
	con.close()

def save_sentiments_to_db(sentiments: list, dbname: str):
	con = sqlite3.connect(dbname)
	cur = con.cursor()
	cur.execute("CREATE TABLE IF NOT EXISTS sentiment_analysis(document_id, sentiment, FOREIGN KEY (document_id) REFERENCES documents(document_id))")
	for i, sentiment in enumerate(sentiments):
		cur.execute("INSERT INTO sentiment_analysis VALUES(?, ?)", (i, sentiment))
	con.commit()
	con.close()

def save_additional_field_to_db(fieldvalues: tuple, dbname: str):
	con = sqlite3.connect(dbname)
	cur = con.cursor()
	cur.execute("CREATE TABLE IF NOT EXISTS additional_fields(text_id, fieldname, fieldvalue, FOREIGN KEY (text_id) REFERENCES documents(text_id))")
	for i, fieldvalue in enumerate(fieldvalues):
		cur.execute("INSERT INTO additional_fields VALUES(?, ?, ?)", (i, fieldvalue[0], fieldvalue[1]))
	con.commit()
	con.close()

def save_topics_to_db(topics: list, dbname: str):
	con = sqlite3.connect(dbname)
	cur = con.cursor()
	cur.execute("CREATE TABLE IF NOT EXISTS lda_topics(topic_id, topic)")
	for i, topic in enumerate(topics):
		cur.execute("INSERT INTO lda_topics VALUES(?, ?)", (i, ' '.join(topic)))
	con.commit()
	con.close()
	
def save_assignments_to_db(assignments: list, dbname: str):
	con = sqlite3.connect(dbname)
	cur = con.cursor()
	cur.execute("CREATE TABLE IF NOT EXISTS assigned_topics(document_id, topic_id, FOREIGN KEY (document_id) REFERENCES documents(document_id), FOREIGN KEY (topic_id) REFERENCES lda_topics(topic_id))")
	
	insert_rows = ()
	for idx, topic_idx in enumerate(assignments):
		cur.execute("INSERT INTO assigned_topics VALUES(?, ?)", (idx, int(topic_idx)))
	con.commit()
	con.close()

def get_topic_assignments_from_db(dbname: str):
	con = sqlite3.connect(dbname)
	cur = con.cursor()
	res = cur.exceute("SELECT documents.document AND lda_topics.topic FROM document, lda_topics, assigned_topics WHERE document.document_id = assigned_topics.document_id AND topics.topic_id = assigned_topics.topic_id")
	return res.fetchall()

	
