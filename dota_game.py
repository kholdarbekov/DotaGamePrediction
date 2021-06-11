import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier  # n_neighbors=3 : Acc=0.72
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


# Import training dataset
with open('trainingdata.txt', 'r') as f:
    data = [x.strip().rstrip().split(',') for x in f.readlines()]

data_train_x = [x[:-1] for x in data]
data_train_y = [int(x[-1]) for x in data]

# create list/dict with heroes
list_heroes = list()
dict_heroes = dict()

for x in data_train_x:
    for y in x:
        list_heroes.append(y)

# Convert hero names to id numbers (str -> int)
list_heroes = list(sorted(set(list_heroes)))
for i in range(len(list_heroes)):
    dict_heroes[list_heroes[i]] = i + 1 # not to start from 0

# create matrix of heroes using id
train_teams_matrix = [[0 for _ in range(10)] for _ in range(len(data_train_x))]
for i in range(len(data_train_x)):
    for j in range(10):
        train_teams_matrix[i][j] = dict_heroes[str(data_train_x[i][j])]

# Import test data
with open('testdata.txt', 'r') as f:
    ntest = f.readline()
    data_test = [x.strip().rstrip().split(',') for x in f.readlines()]

data_test_x = [x[:-1] for x in data_test]
data_test_y = [int(x[-1]) for x in data_test]

test_teams_matrix = [[0 for _ in range(10)] for _ in range(int(ntest))]
for i in range(int(ntest)):
    for j in range(10):
        test_teams_matrix[i][j] = dict_heroes[str(data_test_x[i][j])]


xtrain = np.asarray(train_teams_matrix)
ytrain = np.asarray(data_train_y)
xtest = np.asarray(test_teams_matrix)

# model = DecisionTreeClassifier()
# model = KNeighborsClassifier(n_neighbors=3)
# model = LogisticRegression()
# model = GaussianNB()

model = RandomForestClassifier(n_estimators=100)
model.fit(xtrain, ytrain)

ypred = model.predict(xtrain)
acc = accuracy_score(ytrain, ypred)
print(f'train acc: {acc}')

preds = model.predict(xtest)
data_test_y = np.asarray(data_test_y)
acc2 = accuracy_score(data_test_y, preds)
print(f'test acc: {acc2}')

# print the predicted values
for i in range(0, int(ntest)):
    print(preds[i])
