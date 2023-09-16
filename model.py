import sqlite3
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

class LDA:
	def __init__(self, vectorized_texts: list, feature_names: list, **kwargs: dict):
		self.vectorized_texts = vectorized_texts
		self.feature_names = feature_names
		self.model = LatentDirichletAllocation(**kwargs)
	
	def fit_model(self):
		self.lda_matrix = self.model.fit_transform(self.vectorized_texts)
		self.lda_components = self.model.components_

	def get_topics(self, top_k: int) -> list:
		all_topic_terms = []
	
		for idx, component in enumerate(self.lda_components):
			zipped = zip(self.feature_names, component)
			top_terms_key = sorted(zipped, key = lambda t: t[1], reverse=True)[:top_k]
			top_terms_list = list(dict(top_terms_key).keys())
			all_topic_terms.append(top_terms_list)
		
		return all_topic_terms

	def get_topic_assignments(self) -> list:
		all_topic_assignments = []

		for i in range(self.lda_matrix.shape[0]):
			assign_max_topic = self.lda_matrix[i].argmax()
			all_topic_assignments.append(assign_max_topic)
		
		return all_topic_assignments

	
