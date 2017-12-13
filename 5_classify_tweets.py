import csv
import pandas as pd

with open("data/spectral_clustering_results.csv", 'rb') as f:
    reader = csv.reader(f)
    spec = list(reader)
    spec = spec[0]

with open("data/tfidf_clustering_results.csv", 'rb') as f:
    reader = csv.reader(f)
    tfidf = list(reader)
    tfidf = tfidf[0]

data = pd.read_csv('data/cleansed_data.csv')

for index, row in data.iterrows():
	data.loc[index,'spec_class'] = str(1-float(spec[index]))
	data.loc[index,'tfidf_class'] = tfidf[index]
	
cols = ['text', 'spec_class', 'tfidf_class']
data.to_csv("data/classified_data.csv", index=False, columns=cols, header=cols)




