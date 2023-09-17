import sys
import re
import pandas as pd
import spacy
import eng_spacysentiment
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class Preprocess:
	def __init__(self, texts: list):
		self.texts = texts
		self.nlp_pipeline = spacy.load('en_core_web_sm')
		self.tfidf = TfidfVectorizer()
		self.count_vectorizer = CountVectorizer()
	
	def lemmatize_and_filter(self) -> list:
		sentences = []
		filtered_texts = []
		doc_idx = 0
		print("Preprocessing...")
		for text in tqdm(self.texts):
			try:
				doc = self.nlp_pipeline(text)
			except:
			for sentence in doc.sents:
				sentences.append((doc_idx, sentence.text))
				filtered_tokens = [token.lemma_.lower() for token in sentence if 
								not token.is_punct 
								and not token.is_stop
								and not token.is_currency
								and not token.is_digit
								and not token.like_num
								and not token.is_space
								and token.is_oov
							]
				filtered_texts.append(' '.join(filtered_tokens))
			doc_idx += 1
		self.sentences = sentences
		self.filtered_texts = filtered_texts


	def vectorize_tfidf(self):
		return self.tfidf.fit_transform(self.filtered_texts)

	def vectorize_count(self):
		return self.count_vectorizer.fit_transform(self.filtered_texts)
		
	def get_sentiment(self):
		sentiments = []
		analyzer = eng_spacysentiment.load()
		for idx, sentence in self.sentences:
			analysis = analyzer(sentence)
			sentiment = max(analysis.cats, key=analysis.cats.get)
			sentiments.append(sentiment)
		return sentiments


				


