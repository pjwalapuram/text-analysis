import sys
import re
import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class Preprocess:
	def __init__(self, texts: list):
		self.texts = texts
		self.nlp_pipeline = spacy.load('en_core_web_sm')
		self.tfidf = TfidfVectorizer()
		self.count_vectorizer = CountVectorizer()
	
	def lemmatize_and_filter(self, text: str) -> list:
		doc = self.nlp_pipeline(text)
		filtered_tokens = [token.lemma_.lower() for token in doc if 
				not token.is_punct 
				and not token.is_stop
				and not token.is_currency
				and not token.is_digit
				and not token.like_num
				and not token.is_space
				and token.is_oov
			]
		return ' '.join(filtered_tokens)

	def vectorize_tfidf(self):
		filtered_texts = [self.lemmatize_and_filter(text) for text in self.texts]
		return self.tfidf.fit_transform(filtered_texts)

	def vectorize_count(self):
		filtered_texts = [self.lemmatize_and_filter(text) for text in self.texts]
		return self.count_vectorizer.fit_transform(filtered_texts)
		

	


