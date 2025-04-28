import pandas as pd
import numpy as np
import joblib
import warnings

from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support

from anomaly_detection.data import data_preprocessing

# comment out the next line to see the warning
warnings.filterwarnings('ignore')

class MLModels():

    def __init__(self, data_path: str):
        super(MLModels, self).__init__()
        self.data_path = data_path

    param_grids = {
    'logistic_regression': {
        'estimator': LogisticRegression(solver='lbfgs', max_iter=5000),
        'params': {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l2']
        }
    },
    'random_forest': {
        'estimator': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [100, 300, 1000],
            'max_depth':    [None, 5, 10, 20],
            'max_features': ['sqrt', 'log2']
        }
    },
    'decision_tree': {
        'estimator': DecisionTreeClassifier(random_state=42),
        'params': {
            'max_depth':         [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
        }
    },
    'linear_svm': {
        'estimator': LinearSVC(max_iter=10000),
        'params': {
            'C':    [0.01, 0.1, 1, 10],
            'loss': ['hinge', 'squared_hinge']
            }
        }
    }

    def load_and_preprocess_data(self) -> pd.DataFrame:

        return data_preprocessing.preprocess_data(self.data_path)
    
    def data_split(self):

        df = self.load_and_preprocess_data()
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.fit_transform(X_test)

        joblib.dump(scaler, "./models/scaler.pkl")

        self.X_train, self.X_test = X_train_scaled, X_test_scaled
        self.y_train, self.y_test = y_train, y_test
        self.X_full, self.y_full   = X, y

        return X_train_scaled, X_test_scaled, y_train, y_test
    

    def evaluate(self, estimator, y_pred, y_test, cross_validation_splits=5):
        print(f"Accuracy of the {estimator} model:", (y_pred == y_test).sum() / len(y_pred) * 100, "%")
        print(f"Accuracy with test data {accuracy_score(y_test, y_pred):.2f}")

        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion matrix with test data:\n {cm}")

        precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred)
        print(f"precision: {precision}")
        print(f"recall: {recall}")
        print(f"fscore: {fscore}")

        scores = cross_val_score(estimator, self.X_full, self.y_full, cv=cross_validation_splits)
        print(f"Cross-Validation Accuracy: {np.mean(scores):.2f}")

        cm_for = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_for)
        disp.plot(cmap='Greens')
        plt.title("Confusion Matrix")
        plt.show()

    def all_models_and_parameters(self):
        self.data_split()

        for name, config in self.param_grids.items():
            print(f"Model is {name} and its parameters are {config}")

            grid_search = GridSearchCV(
                estimator = config["estimator"],
                param_grid=config["params"],
                cv=5,
                scoring='f1_macro', #https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
                n_jobs=-1)
            
            grid_search.fit(self.X_train, self.y_train)

            best = grid_search.best_estimator_
            print(f"Best params for {name}: {grid_search.best_params_}")
            print(f"Best CV f1_macro: {grid_search.best_score_:.2f}")

            joblib.dump(best, f"./models/{name}_best.pkl")

            y_pred = best.predict(self.X_test)
            print(f"--- Evaluation for {name} on test set ---")
            self.evaluate(best, y_pred, self.y_test)


    def model_random_forest(self):

        X_train, X_test, y_train, y_test = self.data_split()

        # Random Forest
        rand_for = RandomForestClassifier(n_estimators=1000, max_depth=7, random_state=2024)
        rand_for.fit(X_train, y_train)
        y_pred_rand_for = rand_for.predict(X_test)

        self.evaluate(rand_for, y_pred_rand_for, y_test, cross_validation_splits=5)

        joblib.dump(rand_for, "./models/random_forest_model.pkl")

    def model_svm(self):

        X_train, X_test, y_train, y_test = self.data_split()

        svm = LinearSVC()
        svm.fit(X_train, y_train)

        y_pred_svm = svm.predict(X_test)

        self.evaluate(svm, y_pred_svm, y_test, cross_validation_splits=5)

        joblib.dump(svm, "./models/svm_model.pkl")

    def model_logistic_regression(self):

        X_train, X_test, y_train, y_test = self.data_split()

        logistic_reg = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=10000)
        logistic_reg.fit(X_train, y_train)

        y_pred_log_reg = logistic_reg.predict(X_test)

        self.evaluate(logistic_reg, y_pred_log_reg, y_test, cross_validation_splits=5)

        joblib.dump(logistic_reg, "./models/logistic_regression_model.pkl")

    def model_decision_tree(self):
        
        X_train, X_test, y_train, y_test = self.data_split()

        dec_tree = DecisionTreeClassifier(max_depth=7)
        dec_tree.fit(X_train, y_train)
        y_pred_dec_tree = dec_tree.predict(X_test)

        self.evaluate(dec_tree, y_pred_dec_tree, y_test, cross_validation_splits=5)

        joblib.dump(dec_tree, "./models/decision_tree_model.pkl")

def main():
    data_path = "data/csv_files/combined_with_labels_and_synthetic.csv"
    rf = MLModels(data_path)
    rf.all_models_and_parameters()
    #rf.model_decision_tree()
    #rf.model_logistic_regression()
    #rf.model_svm()
    #rf.model_random_forest()
    #print(df.tail(15))

if __name__ == '__main__':
    main()