import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('./newdata70words.pickle', 'rb'))
data = data_dict['data']
labels = data_dict['labels']

data = np.asarray(data)
labels = np.asarray(labels)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))

f = open('newmodel10words60_70.p', 'wb')
pickle.dump({'model': model}, f)
f.close()



# import pickle
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import LogisticRegression

# # Load data
# data_dict = pickle.load(open('./data.pickle', 'rb'))
# data = np.asarray(data_dict['data'])
# labels = np.asarray(data_dict['labels'])

# # Split dataset
# x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# # List of classifiers
# classifiers = {
#     "Random Forest": RandomForestClassifier(),
#     "Gradient Boosting": GradientBoostingClassifier(),
#     "SVM": SVC(),
#     "K-Nearest Neighbors": KNeighborsClassifier(),
#     "Decision Tree": DecisionTreeClassifier(),
#     "Naive Bayes": GaussianNB(),
#     "Logistic Regression": LogisticRegression()
# }

# # Train and evaluate each classifier
# results = {}
# for name, clf in classifiers.items():
#     clf.fit(x_train, y_train)
#     y_predict = clf.predict(x_test)
#     accuracy = accuracy_score(y_test, y_predict)
#     results[name] = accuracy
#     print(f"{name}: {accuracy * 100:.2f}% accuracy")

# # Save the best performing model
# best_model = max(results, key=results.get)
# print(f"\nBest Model: {best_model} with {results[best_model] * 100:.2f}% accuracy")

# with open('best_model.p', 'wb') as f:
#     pickle.dump({'new_model': classifiers[best_model]}, f)
