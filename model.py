from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

class LDA:
	def __init__(self, vectorized_texts: list, feature_names: list, num_topics: int) -> list:
		self.vectorized_texts = vectorized_texts
		self.feature_names = feature_names
		self.model = LatentDirichletAllocation(n_components=num_topics)
	
	def fit_model(self):
		lda_matrix = self.model.fit_transform(self.vectorized_texts)
		lda_components = self.model.components_
		for idx, component in enumerate(lda_components):
			zipped = zip(self.feature_names, component)
			top_terms_key = sorted(zipped, key = lambda t: t[1], reverse=True)[:10]
			top_terms_list = list(dict(top_terms_key).keys())
			print("Topic " + str(index) + ": ", top_terms_list)

