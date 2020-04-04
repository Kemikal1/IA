import numpy as np
from sklearn import preprocessing
from sklearn import svm
from bag_of_words import *
from sklearn.naive_bayes import ComplementNB
import csv


class KnnClassifier:

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_labels = train_labels

    def classify_data(self, test_data, num_neighbors=3, metric='l2'):

        if (metric == 'l2'):
            distances = np.sqrt(np.sum((self.train_data - test_data) ** 2, axis=1))
        elif (metric == 'l1'):
            distances = np.sum(abs(self.train_data - test_data), axis=1)
        else:
            print('Error! Metric {} is not defined!'.format(metric))

        sort_index = np.argsort(distances)
        sort_index = sort_index[:num_neighbors]
        nearest_labels = self.train_labels[sort_index]
        histc = np.bincount(nearest_labels)

        return np.argmax(histc)

    def classify_datas(self, test_data, num_neighbors=3, metric='l2'):
        num_test_images = test_datashape[0]
        predicted_labels = np.zeros((num_test_images), np.int)

        for i in range(num_test_images):
            if (i % 50 == 0):
                print('processing {}%...'.format(float(i) / num_test_images * 100))
            predicted_labels[i] = self.classify_data(test_data[i, :], num_neighbors=num_neighbors, metric=metric)

        return predicted_labels


def get_id(data):
    ids=[]
    for lines in data:
        lines1=lines.split()
        ids.append(lines1[0])
    return ids
def get_id_label(data):
    ids=[]
    for lines in data:
        ids.append(lines[0])
    return ids
def get_data(data):
    list=[]
    for lines in data:
        lines1=lines.split()
        list.append(lines1)
    return list

def remove_id(data):
    list=[]
    for lines in data:
        lines.pop(0)
        list.append(lines)
    return list
def accuracy(true,pred):
    sum=0
    for i in range(len(true)):
        if true[i]==pred[i]:
            sum+=1
    acc= float(sum)/len(true)
    return float(acc)
def get_labels(data):
    arr=np.zeros(len(data))
    for i in range(len(data)):
        arr[i]=data[i][1]
    return arr
def normalize_data(train_data, test_data, type=None):
    scaler = None
    if type == 'standard':
        scaler = preprocessing.StandardScaler()

    elif type == 'min_max':
        scaler = preprocessing.MinMaxScaler()

    elif type == 'l1':
        scaler = preprocessing.Normalizer(norm='l1')

    elif type == 'l2':
        scaler = preprocessing.Normalizer(norm='l2')

    if scaler is not None:
        scaler.fit(train_data)
        scaled_train_data = scaler.transform(train_data)
        scaled_test_data = scaler.transform(test_data)
        return (scaled_train_data, scaled_test_data)

"""
am importat clasa bag of words pentru reprezentarea frazelor sub o forma de array de 1 si 0
am definit knn classifierul si alte functii pentru a formata datele intrun mod convenabil pentru classifier

si am definit o functie de normalizare

"""
training_data=np.loadtxt('data/train_samples.txt',delimiter='\n',dtype='string')
training_labels=np.loadtxt('data/train_labels.txt')

test_data=np.loadtxt('data/validation_samples.txt',delimiter='\n',dtype='string')
test_labels=np.loadtxt('data/validation_labels.txt')

true_test_data=np.loadtxt('data/test_samples.txt',delimiter='\n',dtype='string')


train_id=[]
test_id=[]
training_labels_id=[]
test_labels_id=[]


training_labels_id=get_id_label(training_labels)
training_labels_id.sort()

test_labels_id=get_id_label(test_labels)
test_labels_id.sort()

true_test_id=get_id(true_test_data)
true_test_id.sort()
true_test_data=get_data(true_test_data)
true_test_data=sorted(true_test_data,key=lambda x:x[0])
true_test_data=remove_id(true_test_data)



train_data_id=get_id(training_data)
train_data_id.sort()


test_data_id=get_id(test_data)
test_data_id.sort()


test_data=get_data(test_data)
test_data=sorted(test_data,key=lambda x:x[0])
test_data=remove_id(test_data)

training_data=get_data(training_data)
training_data=sorted(training_data,key=lambda x:x[0])
training_data=remove_id(training_data)

training_labels=get_labels(training_labels)
training_labels=training_labels.astype(int)
test_labels=get_labels(test_labels)
test_labels=test_labels.astype(int)

""""
am separat id-urile de date si de label-uri

si am transformat in array restul de label-uri
"""



print(test_labels.shape)
print(training_labels)
print("labels",training_labels)
print(training_labels.shape)
bow_model=Bag_of_words()
bow_model.build_vocabulary(training_data)
train_features=bow_model.get_features(training_data)
test_features=bow_model.get_features(test_data)
true_test_features=bow_model.get_features(true_test_data)

"""
folosirea bow pentru a genera features pentru fiecare set de date
"""

scaled_train,scaled_true_test=normalize_data(train_features,true_test_features,'l2')
print(accuracy(test_labels,test_labels))
scaled_train,scaled_test=normalize_data(train_features,test_features,'l1')
scaled_train=np.asarray(scaled_train,dtype=np.float32)
scaled_test=np.asarray(scaled_true_test,dtype=np.float32)
scaled_true_test=np.asarray(scaled_true_test)
"""
normalizarea datelor
"""
print(scaled_test)
print (scaled_train)

print(test_labels)
print(training_labels)
training_labels=np.concatenate((training_labels,test_labels))
scaled_train=np.concatenate((scaled_train,scaled_test))

print("data",scaled_train)
print(scaled_train.shape)

"""
antrenare knn si predictie de etichete
"""


knn=KnnClassifier(scaled_train,training_labels)


predicted_labels=knn.classify_datas(scaled_true_test)
"""""
svm_model=svm.SVC(C=50,kernel='linear')
svm_model.fit(scaled_train,training_labels)
print('fit')
predict1=svm_model.predict(scaled_test)
"""


print(predicted_labels[:100])
print(test_labels[:100])


with open('sub.csv', mode='w') as sub_file:
    sub_writer=csv.writer(sub_file,delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    sub_writer.writerow(['id','label'])
    for i in range(len(true_test_id)):
        sub_writer.writerow([true_test_id[i],predicted_labels[i]])
