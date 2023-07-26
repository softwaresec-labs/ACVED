import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer
import pickle

vulnerability_df = pd.read_pickle("D:\\PhD\\Model_Building\\Stage_6_ML_Model_Training\\Version_3.0\\proccessed_dataset_for_analysis.pickle")

vulnerability_df = vulnerability_df.loc[vulnerability_df['Vulnerability_status'] == 1]
vulnerability_df = vulnerability_df[['processed_code','CWE_ID']]

print(vulnerability_df.CWE_ID.value_counts())

# vulnerability_df = vulnerability_df[vulnerability_df.CWE_ID != ""]
vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("","Other")

print(vulnerability_df.CWE_ID.value_counts())

vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-327","Other")
vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-919","Other")
vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-927","Other")
vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-250","Other")
vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-295","Other")
vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-79","Other")
vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-649","Other")
vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-926","Other")
vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-330","Other")
vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-299","Other")
vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-297","Other")
vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-502","Other")
vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-509","Other")

print(vulnerability_df.CWE_ID.value_counts())

df_79 = c_79 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-79']
df_89 = c_89 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-89']
df_200 = c_200 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-200']
df_250 = c_250 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-250']
df_276 = c_276 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-276']
df_295 = c_295 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-295']
df_297 = c_297 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-297']
df_299 = c_299 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-299']
df_312 = c_312 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-312']
df_327 = c_327 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-327']
df_330 = c_330 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-330']
df_502 = c_502 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-502']
df_532 = c_532 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-532']
df_599 = c_599 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-599']
df_649 = c_649 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-649']
df_676 = c_676 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-676']
df_749 = c_749 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-749']
df_919 = c_919 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-919']
df_921 = c_921 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-921']
df_925 = c_925 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-925']
df_926 = c_926 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-926']
df_927 = c_927 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-927']
df_939 = c_939 = vulnerability_df[vulnerability_df.CWE_ID == 'CWE-939']
df_other = c_other = vulnerability_df[vulnerability_df.CWE_ID == 'Other']

df_532 = c_532.sample(9254)
df_312 = c_312.sample(7649)

vulnerability_df = pd.concat([df_79, df_89,df_200,df_250,df_276,df_295,df_297,df_299,df_312,df_327,df_330,df_502,df_532,df_599,df_649,df_676,df_749,df_919,df_921,df_925,df_926,df_927,df_939,df_other], ignore_index=True)

counts = vulnerability_df.CWE_ID.value_counts()

print(counts)

code_list = vulnerability_df.processed_code.tolist()
y = vulnerability_df.CWE_ID

sentences = code_list
y = y.values

sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1)
vectorizer = CountVectorizer(analyzer = 'word', lowercase=True, max_df=0.80, min_df=10, ngram_range=(1,3))
vectorizer.fit(sentences_train)
X_train = vectorizer.transform(sentences_train)
X_test  = vectorizer.transform(sentences_test)

print(len(vectorizer.vocabulary_))

from sklearn.naive_bayes import MultinomialNB
nb_classifier = MultinomialNB()
nb_model = nb_classifier.fit(X_train, y_train)
nb_predictions = nb_classifier.predict(X_test)
cm = confusion_matrix(y_test, nb_predictions)
print("NB Results")
print(classification_report(y_test, nb_predictions))

from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression(solver='lbfgs', max_iter=1000)
lr_model = lr_classifier.fit(X_train, y_train)
lr_predictions = lr_classifier.predict(X_test)
cm = confusion_matrix(y_test, lr_predictions)
print("LR Results")
print(classification_report(y_test, lr_predictions))

xgb_classifier = XGBClassifier(eval_metric='mlogloss')

from sklearn.preprocessing import LabelEncoder #New version XGB
le = LabelEncoder()
y_train_XGB = le.fit_transform(y_train)

xgb_model = xgb_classifier.fit(X_train, y_train_XGB)
xgb_predictions = xgb_classifier.predict(X_test)
xgb_predictions = le.inverse_transform(xgb_predictions)  #New version XGB

cm = confusion_matrix(y_test, xgb_predictions)
print("XGB Results")
print(classification_report(y_test, xgb_predictions))

from sklearn.ensemble import GradientBoostingClassifier
gb_classifier = GradientBoostingClassifier()
gb_model = gb_classifier.fit(X_train, y_train)
gb_predictions = gb_classifier.predict(X_test)
cm = confusion_matrix(y_test, gb_predictions)
print("GB Results")
print(classification_report(y_test, gb_predictions))

from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier()
rf_model = rf_classifier.fit(X_train, y_train)
rf_predictions = rf_classifier.predict(X_test)
cm = confusion_matrix(y_test, rf_predictions)
print("RF Results")
print(classification_report(y_test, rf_predictions))

from sklearn.neural_network import MLPClassifier
MLP_classifier = MLPClassifier(alpha=1, max_iter=100)
mlp_model = MLP_classifier.fit(X_train, y_train)
mlp_predictions = MLP_classifier.predict(X_test)
cm = confusion_matrix(y_test, mlp_predictions)
print("MLP Results")
print(classification_report(y_test, mlp_predictions))

from sklearn.svm import SVC
SVC_classifier = SVC(gamma=2, C=1)
svc_model = SVC_classifier.fit(X_train, y_train)
svc_predictions = SVC_classifier.predict(X_test)
cm = confusion_matrix(y_test, svc_predictions)
print("SVC Results")
print(classification_report(y_test, svc_predictions))

from sklearn import tree
dt_classifier = tree.DecisionTreeClassifier()
dt_model = dt_classifier.fit(X_train, y_train)
dt_predictions = dt_classifier.predict(X_test)
cm = confusion_matrix(y_test, dt_predictions)
print("DT Results")
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
    ('svc', SVC_classifier),
    ('dt', dt_classifier)

]

stack_model = StackingClassifier(
    estimators=estimator_list, final_estimator=LogisticRegression())

stack_model_new = stack_model.fit(X_train, y_train)
stack_predictions = stack_model_new.predict(X_test)
cm = confusion_matrix(y_test, stack_predictions)
print("ENS Results")
print(classification_report(y_test, stack_predictions))

with open("multiclass_model.pickle", 'wb') as fout:
    pickle.dump((vectorizer, stack_model_new, stack_model), fout)