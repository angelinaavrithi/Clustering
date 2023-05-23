import csv
import string
import nltk
import pandas as pd
import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))
from gensim.models import Word2Vec
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score

# Read dataset
# Convert .txt file to a list of strings
print("\nLOADING: Reading data...")
raw_data = []
with open(filename, newline='') as csvfile:
    file_reader = csv.reader(csvfile, delimiter='\n', quotechar='|')
    for row in file_reader:
        raw_data.append(row[0])

# Text preprocessing
# Remove punctuation and stopwords
print("\nLOADING: Preprocessing utterances...")
clean_data = []
for sent in raw_data:
    words = nltk.word_tokenize(sent) # tokenize words
    sent_no_punct = [word for word in words if word not in string.punctuation] # remove punctuation
    # sent_no_stopwords = [word for word in sent_no_punct if word not in stopwords.words('english')] # remove stopwords
    clean_sentence = ' '.join(sent_no_punct) # join words
    clean_data.append(clean_sentence)

# print(clean_data)

# Text representation using word2vec
# Convert list of string to list of sparse matrices
print("\nLOADING: Vectorizing data...")
word2vec = Word2Vec(sentences=clean_data, vector_size=100, window=5, min_count=1, workers=4)
word_vectors = word2vec.wv #get word vectors
training_set = word_vectors.vectors #create 2d array

# Edit this to change number of clusters
# cluster_number = 7
###

# Cluster and evaluate clustering for 2 to 15 clusters
print("\nLOADING: Clustering data...")
clustering_results=[]
for cluster_number in range(2, 16):
    # Clustering using k-means
    test_kmeans = KMeans(n_clusters=cluster_number, init='k-means++', max_iter=100, n_init=1)
    test_kmeans.fit(training_set)
    clusters = test_kmeans.predict(training_set)

    #test_hier = AgglomerativeClustering(n_clusters=cluster_number, affinity='euclidean', linkage ='ward')
    #clusters = test_hier.fit_predict(training_set)

    #dbscan = DBSCAN(eps=1.5, min_samples=1)
    #clusters = dbscan.fit_predict(training_set)

    #optics = OPTICS()
    #clusters = optics.fit_predict(training_set)

    #brc = Birch(n_clusters=cluster_number)
    #clusters = brc.fit_predict(training_set)

    # Evaluation using silhouette score
    silhouette_avg = silhouette_score(training_set, clusters)
    clustering_results.append((cluster_number, silhouette_avg))

# Print results per cluster number
print("\nLOADING: Printing results...")
print("\n-------------- RESULTS ---------------")
print("Cluster Amount\t\tSilhouette Score")
for result in clustering_results:
    print(f"{result[0]}\t\t\t\t\t{result[1]}")
print("---------------------------------------")

# Edit this to change number of clusters
cluster_number = 4
###

# Perform clustering again using the cluster amount that performed best
print("\nLOADING: Clustering data again...")
final_kmeans = KMeans(n_clusters=cluster_number, init='k-means++', max_iter=100, n_init=1)
final_kmeans.fit(training_set)
final_clusters = final_kmeans.predict(training_set)

# Map utterance to vector
# Save in a dictionary
print("\nLOADING: Mapping vectors to utterances...")
clusters = {}
for utterance, cluster_label in zip(clean_data, final_kmeans.labels_):
    if cluster_label not in clusters:
        clusters[cluster_label] = []
    clusters[cluster_label].append(clean_data)

# Store dictionary in a DataFrame
# Save as a .csv file
print("\nLOADING: Saving clusters to file...")
cluster_table = pd.DataFrame({'Cluster #': [], 'Utterance': []})
for cluster_label, utterance in clusters.items():
    for utterance in clean_data:
        cluster_table = cluster_table.append({'Cluster #': cluster_label, 'Utterance': utterance}, ignore_index=True)

# Save the cluster table to a CSV file
cluster_table.to_csv('cluster_table.csv', index=False)