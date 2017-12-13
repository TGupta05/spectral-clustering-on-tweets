import numpy as np 
import pandas as pd 
import math
import scipy
import scipy.linalg

def create_dictionary():
	dictionary = {}

	for index, row in data.iterrows():
		try:
			unset_tokents = row['tokens'].split(" ")
			unigram = set(unset_tokents)
			bigram = []
			for i in xrange(1, len(unset_tokents)):
				bigram += [unset_tokents[i-1] + " " + unset_tokents[i]]
			bigram = set(bigram)

			for term in unigram:
				if term in dictionary: 
					dictionary[term] += 1
				else: dictionary[term] = 1

			for term in bigram:
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

	row_sums = tf_idf_matrix.sum(axis=1)
	tf_idf_matrix /= row_sums[:, np.newaxis]

	return tf_idf_matrix

def get_similarity_matrix():
	m = np.matmul(tf_idf_matrix, tf_idf_matrix.transpose())
	for i in xrange(size):
		for j in xrange(size):
			d = 1-m[i,j]
			m[i,j] = math.exp(-d/(50*50))
	return m

def get_diag(A):
  return np.diag(A.sum(axis=0))

def get_L(A,D):
  (m,n) = D.shape;
  D_inv = D.copy();
  for i in range(0,m):
    D_inv[i,i] = 1/np.sqrt(D[i,i]);
  return np.matmul(np.matmul(D_inv, A), D_inv);

def get_eigens(L):
  (m,n) = L.shape
  return scipy.linalg.eigh(L, eigvals=(n-2,n-1))

def get_Y(X):
  (m,n) = X.shape
  for i in range(0,m):
    col_sum = math.pow(X[i,0],2) + math.pow(X[i,1],2)
    col_sum = math.sqrt(col_sum)
    X[i,0] /= col_sum
    X[i,1] /= col_sum
  return X

#################################################################################
#################################################################################

data = pd.read_csv('data/cleansed_data.csv')
size = data.shape[0]
dictionary = create_dictionary()
words = dictionary.keys()
word_size = len(words)
words.sort()
inidicies = get_indicies()

print("making tf_idf matrix")
tf_idf_matrix = create_tf_idf_matrix()
# print("saving tf-idf matrix")
# np.savetxt("data/tf_idf_matrix.txt", tf_idf_matrix)

print("making similarity matrix")
similarity_matrix = get_similarity_matrix()
A = np.nan_to_num(similarity_matrix)

print('making diagonal matrix D')
D = get_diag(A)
D = np.nan_to_num(D)

print('making laplacian matrix')
L = get_L(A,D);
L = np.nan_to_num(L)

print('getting eigenvectors')
(eigenvalues,eigenvectors) = get_eigens(L);
eigenvectors = np.nan_to_num(eigenvectors)

print('normalizing matrix')
Y = get_Y(eigenvectors)
Y = np.nan_to_num(Y)
# Y[:,[0, 1]] = Y[:,[1, 0]]
print('saving matrix Y')
np.savetxt("data/Y_matrix.txt", Y)
print("saved!!!")


