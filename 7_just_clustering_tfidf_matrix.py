from sklearn.cluster import KMeans
import numpy as np 
import pandas as pd 
import math

def create_dictionary():
	dictionary = {}

	for index, row in data.iterrows():
		try:
			unset_tokents = row['tokens'].split(" ")
			tokens = set(unset_tokents)
			for term in tokens:
				if term in dictionary: 
					dictionary[term] += 1
				else: dictionary[term] = 1
			for i in xrange(1, len(unset_tokents)):
				term = unset_tokents[i-1] + " " + unset_tokents[i]
				if term in dictionary: 
					dictionary[term] += 1
				else: dictionary[term] = 1
		except: continue
	return dictionary

def get_indicies():
	inidicies = {}
	for i in xrange(0, word_size):
		inidicies[words[i]] = i 
	return inidicies

def create_tf_idf_matrix():
	tf_idf_matrix = np.zeros((size,word_size))

	for index, row in data.iterrows():

		try:
			t = row['tokens'].split(" ")
			tokens = t
			for i in xrange(1, len(t)):
				tokens += [t[i-1] + " " + t[i]]
			s = len(tokens)
			for term in tokens:
				i = inidicies[term]
				idf = math.log(word_size/dictionary[term])
				tf_idf_matrix[index,i] += (1.0 * idf / s)

		except:
			continue

	return tf_idf_matrix

data = pd.read_csv('data/cleansed_data.csv')
size = data.shape[0]
dictionary = create_dictionary()
words = dictionary.keys()
word_size = len(words)
words.sort()
inidicies = get_indicies()
print("making tf_idf matrix")
tf_idf_matrix = create_tf_idf_matrix()
print("performing clustering")
kmeans = KMeans(n_clusters=2)
kmeans.fit(tf_idf_matrix)
y_kmeans = kmeans.predict(tf_idf_matrix)
print(y_kmeans.tolist())


