#Deandra Spike-Madden
#500946801

# ------------------------- IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import regex as re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.stem import PorterStemmer
from wordcloud import WordCloud
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer

nltk.download('wordnet')
nltk.download('omw-1.4')
stopwords = np.loadtxt("cacm_stopwords.txt",dtype=str)
vectorizer = TfidfVectorizer(analyzer='word', norm ='l2',stop_words={'english'})
kos_vocab = np.loadtxt("vocab.kos.txt", dtype=str)
pca = PCA(3)

# --------------------------------------------- PREPROCESSING 
def vectorize_document(documents_dict,vectorizer): # vectorize .....
    X = vectorizer.fit_transform(documents_dict)
    return X

def more_digits(word): #Checks if word contains more letters than numbers
    l = len(word)
    numbers = len("".join(re.findall('[0-9]+', word)))
    letters = len("".join(re.findall('[^0-9]', word)))
    if (numbers/l)<=(letters/l):
        return False
    return True

def build_kos_vocab(kos_vocab): #Removing underscores from vocab then assigning word id to vocab

    build_vocab = []

    for i in range(len(kos_vocab)): # Cleaning vocab
        clean=kos_vocab[i].replace("_"," ").strip()
        build_vocab.append(clean)

    with open('updated_vocab.kos.txt', 'a') as f: # Writing clean vocab to txt
        f.write('\n'.join(build_vocab))
    
    updated_kos_vocab = {}

    with open("updated_vocab.kos.txt",'r') as file_obj:
        words = file_obj.readlines()
        for i in range(len(words)):
            updated_kos_vocab[str(i+1)]=words[i].strip() # Strip to remove new line
    return updated_kos_vocab

#------------Stemming

def build_document(updated_vocab):
    porter_stemmer = PorterStemmer() # to reduce redundancy in documents
 
    doc_word_ids = {}
    # Get word ids per document
    with open("docword.kos.txt") as file:
        for line in file:
            content = line.split(" ")
            if len(content)>1:
                if content[0] in doc_word_ids: doc_word_ids[content[0]].append(content[1])
                else: doc_word_ids[content[0]]=[content[1]]     
            else: continue

    #Construct documents
    kos_documents_ps = []
    for document in doc_word_ids:
        row = []
        for word_id in doc_word_ids[document]:
            word = updated_vocab[word_id]
            if word in stopwords:continue
            if more_digits(word): continue # more letters in word than numbers
            stemmed = porter_stemmer.stem(word) # Stem words using porter algorithm
            if stemmed in row: continue # Remove redundancy in documents
            row.append(stemmed)
        kos_documents_ps.append(" ".join(row))
    return kos_documents_ps

# --------------------------------------------- OPTIMAL K VALUE
def find_optimal_kV1(min,max,vec_documents):
    model = KMeans(init = 'k-means++', max_iter = 300, n_init = 10)  # k is range of number of clusters.
    visualizer = KElbowVisualizer(model, k=(min,max), timings=False)
    visualizer.fit(vec_documents)        # Fit the data to the visualizer
    visualizer.show() 

#--------------------------------------------- TESTING KMEANS CLUSTERS
def kmeans(document_vec,k,pca,type):
    terms_in_cluster = {}
    model = KMeans(n_clusters= k, init='k-means++', random_state=42)     # Creating Model
    
    df = pca.fit_transform(document_vec.todense())  #Transform the data
    y_model =  model.fit(document_vec) #to be able to retrieve each term

    print("Inertia for model with " +str(k)+ " clusters is: " + str(y_model.inertia_))

    print("Top 40 terms in each cluster (K="+str(k)+","+type+")")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out ()

    for i in range(k):
        cloud_data = ""
        clus = []

        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :40]:
            print(' %s' % terms[ind], end='')
            cloud_data+=terms[ind] + " "
            clus.append(terms[ind])
        terms_in_cluster[i]=clus
        print()
        print()
        word_cloud = WordCloud(collocations = False, background_color = 'white',colormap='tab20',).generate(cloud_data)
        plt.imshow(word_cloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
    
    y_model =  model.fit(df) # graphing clusters 
    labels_model = y_model.labels_

    plt.figure(figsize=(20,10)) #Plot clusters in 2d

    #Getting the Centroids
    centroids = model.cluster_centers_
    u_labels = np.unique(labels_model)
    
    #plotting the results:
    for i in u_labels:
        plt.scatter(df[labels_model == i , 0] , df[labels_model == i , 1] , label = i,s=15)
    plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
    plt.title("Kmeans Clusters - 2D (K="+str(k)+","+type+")",fontsize=14)
    plt.legend(fontsize=20)
    plt.show()

    fig = plt.figure(figsize=(20,10))  #Plot clusters 3d
    ax = fig.add_subplot(111, projection='3d')
    
    #plotting the results:
    for i in u_labels:
        ax.scatter(df[labels_model == i , 0] , df[labels_model == i , 1],  df[labels_model == i , 2],label = i,s=8)
    ax.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
    plt.title("Kmeans Clusters - 2D (K="+str(k)+","+"Stemmed"+")",fontsize=14)
    plt.legend(fontsize=13)
    plt.show()
    return [model,terms_in_cluster]

#---------------------------------------------MAIN
updated_kos_vocab = build_kos_vocab(kos_vocab)
documents = build_document(updated_kos_vocab)
vec_kos_documents= vectorize_document(documents,vectorizer)

find_optimal_kV1(3,16,vec_kos_documents) # finding optimal k using package (Elbow Method)

print(str(8)+ " is the suggest number of clusters using the elbow method. \n\nI will be investigating when k = 8, 6, and 4")

print("Let's start off with k = 8.")
k8 = kmeans(vec_kos_documents,8,pca,"Stemmed")

print("When k = 6.")
k6 = kmeans(vec_kos_documents,6,pca,"Stemmed")

print("When k = 4.")
k4 = kmeans(vec_kos_documents,5,pca,"Stemmed")


