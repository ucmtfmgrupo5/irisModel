# Load libraries
import pickle
from pandas import read_csv
from pandas.core.frame import DataFrame
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import os


# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)
# Spot Check Algorithms

model = LogisticRegression(solver='liblinear', multi_class='ovr')
model.fit(X_train, Y_train)

# Save model

currrentPath = os.path.dirname(os.path.abspath(__file__))
parentPath = os.path.abspath(os.path.join(currrentPath, os.pardir))


filename = parentPath+'/modelSampleModel/modelSampleModel'
outfile = open(filename,'wb')

pickle.dump(model, outfile)

outfile.close()






