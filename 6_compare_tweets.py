import pandas as pd 

data = pd.read_csv('data/manually_classified.csv')
numread=500
s_count = 0
t_count = 0
for index, row in data.iterrows():
	if index >= numread: break
	if row['manual'] == float(row['spec_class']): s_count += 1
	if row['manual'] == float(row['tfidf_class']): t_count += 1

print("spectral clustering result = " + str(s_count/float(numread)))
print("k-means clustering result = " + str(t_count/float(numread)))
