# TODO convert:
# ethnicity => 7 categories
# gender => 2 categories
# hospital_admit_source => 16 categories
# icu_admit_source => 6 categories
# icu_stay_type => 3 categories
# icu_type => 8 categories
# apache_3j_bodysystem => 12
# apache_2_bodysystem => 11
# readmission_status => don't need as it is all 0
# weight => delete NA
# df = df.loc[:, df.isin([' ','NA', 0]).mean() < .6]

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from definitions import TRAIN_RAW, TEST_RAW
from definitions import TRAIN_PROCESS, TEST_PROCESS
from definitions import SOLUTION_TEMPLATE

class MultiLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col].astype('str'))
        else:
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col.astype('str'))
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def df_has_only_numeric_columns(df):
    assert df.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()).all()


def preprocess_data(Train=True):
    if Train:
        df_tr = pd.read_csv(TRAIN_RAW)
    else:
        df_tr = pd.read_csv(TEST_RAW)

    df_train = df_tr.copy()

    # drop readmission_status as it has all 0
    df_train.drop('readmission_status', axis=1, inplace=True)
    df_train.drop('encounter_id', axis=1, inplace=True)
    df_train.drop('patient_id', axis=1, inplace=True)
    df_train.drop('hospital_id', axis=1, inplace=True)
    df_train.drop('icu_id', axis=1, inplace=True)

    # 25 genders are nan => replace them with 'M'
    df_train['gender'] = df_train['gender'].fillna(df_train['gender'].value_counts().index[0])
    print(df_train['gender'].isnull().sum())
    df_train['gender'] = df_train['gender'].map({'M': 0, 'F': 1})

    df_train['icu_stay_type'] = df_train['icu_stay_type'].fillna(df_train['icu_stay_type'].value_counts().index[0])
    print(df_train['icu_stay_type'].isnull().sum())
    df_train['icu_stay_type'] = df_train['icu_stay_type'].map({'admit': 0, 'transfer': 1})

    und_diag = {"Undefined diagnoses": "Undefined Diagnoses"}
    df_train["apache_2_bodysystem"].replace(und_diag, inplace = True)

    cat_columns = ["ethnicity", "hospital_admit_source",
                   "icu_admit_source", "icu_type",
                   "apache_3j_bodysystem", "apache_2_bodysystem"]

    df_train = MultiLabelEncoder(columns=cat_columns).fit_transform(df_train)

    df_train.fillna(df_train.mean(), inplace=True)

    y = df_train[['hospital_death']]
    df_train.drop('hospital_death', axis=1, inplace=True)

    df_has_only_numeric_columns(df_train)

    X = df_train
    return X, y

X, y = preprocess_data(Train=True)
X_test, y_test = preprocess_data(Train=False)

# model = LinearDiscriminantAnalysis()
#kfold = StratifiedKFold(n_splits=5)
#result = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
#predictions = cross_val_predict(model, X, y, cv=kfold)
#print(result.mean())

#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

rf_clf = RandomForestClassifier()
#grid_clf = GridSearchCV(rf_clf, param_grid, cv=10)
#grid_clf.fit(X_train, y_train.values.ravel())
#print(grid_clf.best_params_)

rf_clf.fit(X, y)
#rf_pred = rf_clf.predict(X_test)
#print(confusion_matrix(y_test, rf_pred))
#print(accuracy_score(y_test, rf_pred))

rf_prob = rf_clf.predict_proba(X_test)
rf_prob = [p[1] for p in rf_prob]

df_solution = pd.read_csv(SOLUTION_TEMPLATE)
df_solution["hospital_death"] = rf_prob
df_solution.to_csv(SOLUTION_TEMPLATE, index=False)
