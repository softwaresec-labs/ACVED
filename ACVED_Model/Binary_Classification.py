import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import math

vulnerability_df = pd.read_pickle("proccessed_dataset_for_analysis.pickle")

print(vulnerability_df.head())

print(vulnerability_df.Vulnerability_status.value_counts())

c_0 = vulnerability_df[vulnerability_df.Vulnerability_status == 0]
c_1 = vulnerability_df[vulnerability_df.Vulnerability_status == 1]

c_0_count = c_0.processed_code.count()
c_1_count = c_1.processed_code.count()

min_count = 0

if(c_0_count<=c_1_count):
    min_count = c_0_count
else:
    min_count = c_1_count

i = (math.ceil(min_count / 1000) * 1000)-1000
print(min_count,i)

df_0 = c_0.sample(i)
df_1 = c_1.sample(i)

vulnerability_df = pd.concat([df_0, df_1], ignore_index=True)

print(vulnerability_df.Vulnerability_status.value_counts())

code_list = vulnerability_df.processed_code.tolist()
y = vulnerability_df.Vulnerability_status

sentences = code_list
y = y.values

sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1)
vectorizer = CountVectorizer(analyzer = 'word', lowercase=True, max_df=0.80, min_df=100, ngram_range=(1,3))
vectorizer.fit(sentences_train)
X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)

print(len(vectorizer.vocabulary_))

from sklearn.naive_bayes import MultinomialNB
nb_classifier = MultinomialNB()
nb_model = nb_classifier.fit(X_train, y_train)
nb_predictions = nb_classifier.predict(X_test)
cm = confusion_matrix(y_test, nb_predictions)
print("NB Results:")
print(classification_report(y_test, nb_predictions))

from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression(solver='lbfgs', max_iter=1000)
lr_model = lr_classifier.fit(X_train, y_train)
lr_predictions = lr_classifier.predict(X_test)
cm = confusion_matrix(y_test, lr_predictions)
print("LR Results:")
print(classification_report(y_test, lr_predictions))

from sklearn.preprocessing import LabelEncoder #New version XGB
le = LabelEncoder()
y_train_XGB = le.fit_transform(y_train)

xgb_classifier = XGBClassifier()
xgb_model = xgb_classifier.fit(X_train, y_train_XGB)
xgb_predictions = xgb_classifier.predict(X_test)
xgb_predictions = le.inverse_transform(xgb_predictions)  #New version XGB

cm = confusion_matrix(y_test, xgb_predictions)
print("XGB Results:")
print(classification_report(y_test, xgb_predictions))

from sklearn.ensemble import GradientBoostingClassifier
gb_classifier = GradientBoostingClassifier()
gb_model = gb_classifier.fit(X_train, y_train)
gb_predictions = gb_classifier.predict(X_test)
cm = confusion_matrix(y_test, gb_predictions)
print("GB Results:")
print(classification_report(y_test, gb_predictions))

from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier()
rf_model = rf_classifier.fit(X_train, y_train)
rf_predictions = rf_classifier.predict(X_test)
cm = confusion_matrix(y_test, rf_predictions)
print("RF Results:")
print(classification_report(y_test, rf_predictions))

from sklearn.neural_network import MLPClassifier
MLP_classifier = MLPClassifier(alpha=1, max_iter=100)
mlp_model = MLP_classifier.fit(X_train, y_train)
mlp_predictions = MLP_classifier.predict(X_test)
cm = confusion_matrix(y_test, mlp_predictions)
print("MLP Results:")
print(classification_report(y_test, mlp_predictions))

# from sklearn.svm import SVC
# SVC_classifier = SVC(gamma=2, C=1)
# svc_model = SVC_classifier.fit(X_train, y_train)
# svc_predictions = SVC_classifier.predict(X_test)
# cm = confusion_matrix(y_test, svc_predictions)
# print("SVC Results:")
# print(classification_report(y_test, svc_predictions))

from sklearn import tree
dt_classifier = tree.DecisionTreeClassifier()
dt_model = dt_classifier.fit(X_train, y_train)
dt_predictions = dt_classifier.predict(X_test)
cm = confusion_matrix(y_test, dt_predictions)
print("DT Results:")
print(classification_report(y_test, dt_predictions))

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

estimator_list = [
    ('nb', nb_classifier),
    ('lr', lr_classifier),
    ('xgb', xgb_classifier),
    ('gb', gb_classifier),
    ('rf', rf_classifier),
    ('mlp', MLP_classifier),
    # ('svc', SVC_classifier),
    ('dt', dt_classifier)

]

stack_model = StackingClassifier(
    estimators=estimator_list, final_estimator=LogisticRegression())

stack_model_new = stack_model.fit(X_train, y_train)
stack_predictions = stack_model_new.predict(X_test)
cm = confusion_matrix(y_test, stack_predictions)
print("ENS Results:")
print(classification_report(y_test, stack_predictions))

with open("binary_model.pickle", 'wb') as fout:
    pickle.dump((vectorizer, stack_model_new, stack_model), fout)