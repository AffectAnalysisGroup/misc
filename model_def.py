from sklearn import svm
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report
import numpy as np

np.random.seed(1983)


# def split_data(data, labels, train_val_test=False, nfolds=3):
#
#     skf_train_test = StratifiedKFold(nfolds, shuffle=True)
#     train_idx, _test_idx = skf_train_test.split(data, labels)
#     train = (data[train_idx, :], labels[train_idx, :])
#     val = (data[_test_idx, :], labels[_test_idx, :])
#
#     if train_val_test:
#
#         data= np.delete(data, train_idx, axis=0)
#         labels = np.delete(labels, train_idx, axis=0)
#
#         skf_val_test = StratifiedKFold(nfolds=2)
#         val_idx, test_idx = skf_val_test(data, labels)
#
#         val = (data[val_idx, :], labels[val_idx])
#         test = (data[test_idx, :], labels[test_idx])
#
#         return train, val, test
#     else:
#         return train, val


class Classifier:

    def __init__(self, basic_model, parameters=None):

        self.basic_model = basic_model

        if parameters is not None:
            self.c_range = parameters['c_range']
            self.gamma_range = parameters['g_range']
        else:
            self.c_range = 13
            self.gamma_range = 13

        self.parameters = {'kernel': ('linear', 'rbf'), 'C': np.logspace(-2, 10, self.c_range), 'gamma':np.logspace(-9, 3, self.gamma_range)}

    def normalize(self, data, labels):

        self.data = np.array(data)
        self.labels = np.array(labels)

        scaler = StandardScaler()
        scaler.fit_transform(self.data)

        return

    def split_data(self, train_val_test=False, nfolds=3):

        data, labels = self.data, self.labels

        train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=1/nfolds, shuffle=True, stratify=labels)
        # train_idx, _test_idx = skf_train_test.split(data, labels)
        train = (train_data, train_labels)
        test = (test_data, test_labels)

        if train_val_test:

            train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=1 / nfolds,
                                                                                shuffle=True, stratify=labels)

            # skf_val_test = StratifiedKFold(nfolds=2)
            # val_idx, test_idx = skf_val_test(data, labels)

            val = (val_data, val_labels)
            train = (train_data, train_labels)

            self.train = train
            self.val = val
            self.test = test

            return train, val, test
        else:

            self.train = train
            self.test = test

            return train, test

    def classify_and_predict(self, folds=3, grid_verbose=True, njobs=-1):

        self.model = GridSearchCV(self.basic_model, self.parameters, cv=folds, verbose=grid_verbose, n_jobs=njobs)
        print('Training GridSearch based SVM')
        self.model.fit(self.train[0], self.train[1])
        # TODO pick the best model to predict
        self.probs = self.model.predict_proba(self.test[0])
        self.pred = self.model.predict(self.test[0])

        return

    def metrics(self):

        acc = accuracy_score(self.test[1], self.pred)
        F_measure = classification_report(self.test[1], self.pred, output_dict=True)
        conf_matrix = confusion_matrix(self.test[1], self.pred)
        auc = roc_auc_score(self.test[1], self.probs[:, -1])

        return {'accuracy':acc, 'f1':F_measure, 'confusion':conf_matrix, 'auc':auc}