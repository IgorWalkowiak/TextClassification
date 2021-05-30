from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn.model_selection import train_test_split


datasetNames = ["Dennis+Schwartz", "James+Berardinelli", "Scott+Renshaw", "Steve+Rhodes"]

def analyzeData(X, labels):

    X_0 = ""
    X_1 = ""
    X_2 = ""
    lens = {"0\n": 0, "1\n": 0, "2\n": 0}

    for x, label in zip(X, labels):
        lens[label] += len(x)

        if label == "0\n":
            X_0+=x
        elif label == "1\n":
            X_1+=x
        elif label == "2\n":
            X_2+=x

    print("Top najczestsze slowa swiadczace o negatywnej recenzji")
    print(Counter(X_0.split()).most_common(100))
    print("Top najczestsze slowa swiadczace o neutralnej recenzji")
    print(Counter(X_1.split()).most_common(100))
    print("Top najczestsze slowa swiadczace o pozytywnej recenzji")
    print(Counter(X_2.split()).most_common(100))
    print()
    print("Top najrzadsze slowa swiadczace o negatywnej recenzji")
    print(list(Counter(X_0.split()))[-10:])
    print("Top najrzadsze slowa swiadczace o neutralnej recenzji")
    print(list(Counter(X_1.split()))[-10:])
    print("Top najrzadsze slowa swiadczace o pozytywnej recenzji")
    print(list(Counter(X_2.split()))[-10:])
    print()

    print("Srednia dlugosc negatywnych recenzji: ", lens["0\n"]/labels.count("0\n"))
    print("Srednia dlugosc neutralnych recenzji: ", lens["1\n"]/labels.count("1\n"))
    print("Srednia dlugosc pozytywnych recenzji: ", lens["2\n"]/labels.count("2\n"))

    print("Ilość negatywnych recenzji: ", labels.count("0\n"))
    print("Ilość neutralnych recenzji: ", labels.count("1\n"))
    print("Ilość pozytywnych recenzji: ", labels.count("2\n"))





def predict(text, model):
    print(model.predict_log_proba([text]))

def getData(datasetName):
    with open('scaledata/' + datasetName + '/subj') as f:
        X = f.readlines()

    with open('scaledata/' + datasetName + '/label.3class') as f:
        Y = f.readlines()
    return X, Y

def getDataFromAllSets():
    X = []
    Y = []
    for x in range(4):
        newX, newY = getData(datasetNames[x])
        X.extend(newX)
        Y.extend(newY)
    return X,Y

vectorizer = CountVectorizer()
NB = MultinomialNB()
model = make_pipeline(vectorizer, NB)
X, y = getDataFromAllSets()
analyzeData(X,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

model.fit(X_train, y_train)


labels = model.predict(X_test)
mat = confusion_matrix(y_test, labels)
fig, ax = plot_confusion_matrix(conf_mat=mat)
plt.show()

