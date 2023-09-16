import sys
import re
import pandas as pd
import spacy

class Preprocess:
	def __init__(self):
		self.nlp_pipeline = spacy.load('en_core_web_sm')
	
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



