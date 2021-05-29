from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

datasetNames = ["Dennis+Schwartz", "James+Berardinelli", "Scott+Renshaw", "Steve+Rhodes"]


model = make_pipeline(TfidfVectorizer(), MultinomialNB())
with open('scaledata/'+datasetNames[0]+'/subj') as f:
    data = f.readlines()

with open('scaledata/'+datasetNames[0]+'/label.3class') as f:
    target = f.readlines()

for x, y in zip(data, target):
    print(y, x)
model.fit(data, target)
labels = model.predict(data)
print(model.predict([data[-1]]))

#print("target",target)
#print("labels",labels)

mat = confusion_matrix(target, labels)
print(mat)
fig, ax = plot_confusion_matrix(conf_mat=mat)
plt.show()
#sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)

