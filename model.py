from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

class LDA:
	def __init__(self, texts: list, num_topics: int):
		self.texts = texts
		self.tfidf = TfidfVectorizer()
		self.count_vectorizer = CountVectorizer()
		self.model = LatentDirichletAllocation(n_components=num_topics)
	
	def vectorize_count(self):
		self.vectorized_texts = self.count_vectorizer.fit_transforme(self.texts)

	def vectorize_tfidf(self):
		self.vectorized_texts = self.tfidf.fit_transform(self.texts)
	
	def get_topics(self):
		lda_matrix = self.model.fit_transform(self.vectorized_texts)
		lda_components = self.model.components_
		terms = self.tfidf.get_feature_names_out()
		for idx, component in enumerate(lda_components):
			zipped = zip(terms, component)
			top_terms_key = sorted(zipped, key = lambda t: t[1], reverse=True)[:10]
			top_terms_list = list(dict(top_terms_key).keys())
			print("Topic " + str(index) + ": ", top_terms_list)

