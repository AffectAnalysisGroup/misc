from sklearn import svm
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_auc_score, classification_report, cohen_kappa_score, f1_score, make_scorer
import numpy as np

np.random.seed(1983)
import warnings
warnings.filterwarnings('ignore')

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
            self.c_range = np.logspace(-2, 0.1, parameters['c_range'])
            self.gamma_range = np.logspace(-2, 0.1,parameters['g_range'])
        else:
            self.c_range = np.logspace(-2, 0.1, 13)
            self.gamma_range = np.logspace(-2, 0.1, 13)

        # self.c_range = [0, 1]
        # self.gamma_range = [0, 1]

        self.parameters = {'kernel': ('linear', 'rbf'), 'C': self.c_range, 'gamma':self.gamma_range}

        kappa = make_scorer(cohen_kappa_score)
        acc = make_scorer(balanced_accuracy_score)
        f1 = make_scorer(f1_score)
        auc = make_scorer(roc_auc_score, average='weighted', needs_proba=True)
        def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
        def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
        def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
        def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]

        self.scores = {'kappa':kappa, 'acc':acc, 'f1':f1, 'auc':auc, 'tp':make_scorer(tp), 'fp':make_scorer(fp)
            , 'tn':make_scorer(tn), 'fn':make_scorer(fn)}

    def normalize(self, data, labels, print_dist = False):

        self.data = np.array(data)
        self.labels = np.array(labels)
        self.labels_dist = []

        if print_dist:
            for lab in np.unique(self.labels):
                self.labels_dist.append(len(np.where(self.labels==lab)[0])/len(self.labels))
            print('data distribution-', self.labels_dist)

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

    def classify_and_predict(self, folds=3, optimize_for = 'auc', grid_verbose=True, njobs=-1):

        self.model = GridSearchCV(self.basic_model, self.parameters, cv=folds, verbose=grid_verbose
                                  , n_jobs=njobs, refit=optimize_for, scoring=self.scores)
        print('Training GridSearch based SVM')
        self.model.fit(self.train[0], self.train[1])
        # TODO pick the best model to predict
        # 	best_idx = model.model.best_index_
        # 	results = model.model.cv_results_
        self.probs = self.model.predict_proba(self.test[0])
        self.pred = self.model.predict(self.test[0])

        return

    def metrics(self):


        # acc and auc are weighted based on the inverse prevalence of each class
        # weighted F1 is chosen from the report?
        print('GT distribution-', np.where(self.test[1]==0)[0].shape, np.where(self.test[1]==1)[0].shape)
        acc = balanced_accuracy_score(self.test[1], self.pred)
        F_measure = classification_report(self.test[1], self.pred, output_dict=True)
        conf_matrix = confusion_matrix(self.test[1], self.pred)
        auc = roc_auc_score(self.test[1], self.probs[:, -1], average='weighted')
        kappa = cohen_kappa_score(self.pred, self.test[1])
        print('best parameters-', self.model.best_params_)

        return {'accuracy':acc, 'f1':F_measure, 'confusion':conf_matrix, 'auc':auc, 'kappa':kappa}