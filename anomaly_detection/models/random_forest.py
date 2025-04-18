import pandas as pd
import numpy as np
import joblib

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support

from anomaly_detection.data import data_preprocessing

class RandomForestModel:

    def __init__(self, data_path: str):

        self.data_path = data_path

    def load_and_preprocess_data(self) -> pd.DataFrame:

        data = data_preprocessing.preprocess_data(self.data_path)

        return data
    
    def data_split(self):
        dataframe = self.load_and_preprocess_data()
        print(dataframe)
        array = dataframe.values

        print(array)

        self.X = array[:, :-1]
        self.y = array[:, -1]
        print(self.y[-1000:])

        # Sugrupuojame i train ir test

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.fit_transform(X_test)

        joblib.dump(scaler, "./models/random_forest_scaler.pkl")

        print(X_test_scaled)
        print(X_train_scaled)
        print(y_test)
        print(y_train)

        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def apskaiciuoti_tikslumo_metrikas(self, estimator, y_pred, y_test, cross_validation_splits=5):
        print(f"Tikslumas apskaiciuotas rankiniu budu {estimator} modelio:", (y_pred == y_test).sum() / len(y_pred) * 100, "%")
        print(f"Accuracy with test data {accuracy_score(y_test, y_pred):.2f}")

        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion matrix with test data:\n {cm}")

        precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred)
        print(f"precision: {precision}")
        print(f"recall: {recall}")
        print(f"fscore: {fscore}")

        scores = cross_val_score(estimator, self.X, self.y, cv=cross_validation_splits)
        print(f"Cross-Validation Accuracy: {np.mean(scores):.2f}")

    def model_random_forest(self):

        X_train, X_test, y_train, y_test= self.data_split()

        # Random Forest
        rand_for = RandomForestClassifier(n_estimators=1000, max_depth=7, random_state=2024)
        rand_for.fit(X_train, y_train)
        y_pred_rand_for = rand_for.predict(X_test)

        self.apskaiciuoti_tikslumo_metrikas(rand_for, y_pred_rand_for, y_test, cross_validation_splits=5)

        joblib.dump(rand_for, "./models/random_forest_model.pkl")

        cm_for = confusion_matrix(y_test, y_pred_rand_for)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_for)
        disp.plot(cmap='Greens')
        plt.title("Confusion Matrix Random Forest")
        plt.show()


def main():
    data_path = "data/csv_files/combined_with_labels_and_synthetic.csv"
    rf = RandomForestModel(data_path)

    rf.model_random_forest()
    #print(df.tail(15))

if __name__ == '__main__':
    main()