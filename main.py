from sklearn.ensemble import AdaBoostClassifier

import os
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import skimage
import seaborn as sns
IMREAD= skimage.io.imread 
def train_ada_boost(X,y,estimators,rand_seed):

    X = np.array([1,34,455,8,-34,567,67,45,654,3,2,1,7,8,9,0,0,0,6,6])#.reshape([20,1])
    print(X.shape)
    if len(X.shape)==1 : X = X.reshape([X.shape[0],1])
    y=np.array([1,0,0,0,1,0,0,0,1,1,1,1,0,0,0,0,0,1,1,1]).reshape([20,])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=3)

    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(X_train,y_train)
    scores = clf.score(X_test,y_test)
    print(scores)
    fig = plt.figure()
    print(np.arange(0,19,1))
    ax = fig.add_subplot()
    ax.scatter(np.arange(0,19,1)[:X_test.shape[0]],X_test, c = clf.predict(X_test))
    ax.scatter(np.arange(0,19,1)[:X_test.shape[0]],X_test, c = y_test, marker = "x")

    plt.show()
    return clf, X_train,y_train
def get_image_collection(path:str, p_h = 1):
    print(path)
    # print(IMREAD(path + "/"+os.listdir(path)[0],as_gray = True).shape)
    collection = np.array([(np.array(IMREAD(path + "/"+f ,as_gray=True)[:680,:1258])).reshape(-1) for n, f in enumerate(os.listdir(path)) if (n%p_h==0)])
   
    return collection

class set_loader:
    def __init__(self):
        self.data = np.array([])
        self.classes = []
        self.datasets = {}

    def load(self,path:str, num):
        for f in os.listdir(path): self.classes.append(str(f))
        
        for f in os.listdir(path):
            p_h = len(os.listdir(path+ '/' + f))//(num-1)
            data = get_image_collection(path+ '/' + f, p_h)
            self.datasets[str(f)]=data
        self.data = np.concatenate([self.datasets[f] for f in self.classes], axis = 0)
    def train_PCA(self):
        pca = PCA(n_components = 3)
        pca.fit(self.data)
        return pca


        

if __name__ == "__main__":
    # collection = get_image_collection("D:/Steven/EmperorEngeneerUniversity/cath/trg_dataset/Audi")
    # fig = plt.figure()

    # ax = fig.add_subplot()
    # print(collection.shape)
    # plt.imshow(collection[0].reshape(680,1358))
    # plt.show()
    sample = 17
    classes = set_loader()
    classes.load("D:/Steven/EmperorEngeneerUniversity/cath/trg_dataset",sample)
    print(classes.classes)
    pca = classes.train_PCA()
    fig = plt.figure()
    ax = fig.add_subplot()
    projected = classes.data.dot(pca.components_.T)
    colors = []
    for i in range(len(classes.classes)):colors.append([i for _ in range(sample)]) 
    print(colors)
    ax.scatter(projected[:,0],projected[:,1],c= colors)
    plt.show()