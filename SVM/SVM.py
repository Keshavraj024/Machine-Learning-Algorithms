"""
Program to train the classifier user Support Vector Machine
"""
from sklearn import datasets, model_selection, svm, metrics
import pickle
import logging


class SVMClassifier:
    def __init__(self, save=False):
        self.data = datasets.load_breast_cancer()
        self.X = self.data.data
        self.Y = self.data.target
        self.save = save

    def train_test_splitter(self) -> tuple:
        """Split the data into train and test

        Returns:
                _type_: tuple
        """
        (
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test,
        ) = model_selection.train_test_split(self.X, self.Y, test_size=0.1)

        return self.x_train, self.x_test, self.y_train, self.y_test

    def train(self) -> svm:
        """Train the model

        Returns:
                _type_: SVMClassifier
        """
        train_x, _, train_y, _ = self.train_test_splitter()
        self.classifier = svm.SVC(kernel="poly")
        self.classifier.fit(train_x, train_y)
        return self.classifier

    def score(self) -> float:
        """Calculate the accuracy of the model

        Returns:
                _type_: float
        """
        _, test_x, _, test_y = self.train_test_splitter()
        y_pred = self.train().predict(test_x)
        accuracy = metrics.accuracy_score(y_pred, test_y)
        return accuracy

    def display_score(self):
        """Display the results"""
        logging.info(f"Accuracy : {self.score()}")

    def save_model(self):
        """Save the model"""
        if self.save:
            with open("svm_model.pickle", "wb") as file:
                pickle.dump(self.train(), file)
            logging.info("Model saved to disk")


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    classifier = SVMClassifier(save=True)
    classifier.display_score()
    classifier.save_model()
