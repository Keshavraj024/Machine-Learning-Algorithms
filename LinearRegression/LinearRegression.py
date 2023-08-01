"""
Program to train the Linear model
"""
import pandas as pd
from sklearn import model_selection, linear_model
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
import logging

class LinearRegression:
    def __init__(self, data, train):
        self.data = pd.read_csv(data, sep=";")
        self.train = train

    def train_test_split(self):
        """Split the data into train an test set

        Returns:
                _type_: traun and test numpy arrays
        """
        self.x = np.array(self.data[["G1", "G2", "absences", "failures", "studytime"]])
        self.y = np.array(self.data[["G3"]])
        (
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test,
        ) = model_selection.train_test_split(self.x, self.y, test_size=0.1)
        return self.x_train, self.x_test, self.y_train, self.y_test

    def fit(self)->linear_model:
        """Train the linear model

        Returns:
                _type_: Linearregressor
        """
        train_x, _, train_y, _ = self.train_test_split()
        self.regressor = linear_model.LinearRegression()
        self.regressor.fit(train_x, train_y)
        return self.regressor

    def save_load_model(self):
        """Save the model"""
        if self.train == True:
            self.regressor = self.fit()
            with open("Model/stdmodel.pickle", "wb") as file:
                pickle.dump(self.regressor, file)
            logging.info("Model is saved")
        else:
            with open("Model/stdmodel.pickle", "rb") as file:
                self.model = pickle.load(file)
            logging.info("Model is loaded")
            self.display_score()

    def score(self)->float:
        """Get the accuracy of the model

        Returns:
                _type_: number
        """
        _, test_x, _, test_y = self.train_test_split()
        accuracy = self.model.score(test_x, test_y)
        return accuracy

    def model_parameters(self)->tuple:
        """Get the model parameters

        Returns:
                _type_: tuple
        """
        coefficients = self.model.coef_
        intercepts = self.model.intercept_
        return coefficients, intercepts

    def display_score(self):
        """Print Everything to Screen"""
        logging.info(f"Accuracy     :  {self.score()}")
        logging.info(f"coefficients : {self.model_parameters()[0]}")
        logging.info(f"Intercepts   :  {self.model_parameters()[1]}")

    def data_visualize(self, visualize):
        """
        Visualize data
        Args:
                visualize (_type_): bool
        """
        if visualize == True:
            style.use("ggplot")
            plt.scatter(self.data[["G1"]], self.data[["G2"]])
            plt.xlabel("G1")
            plt.ylabel("G3")
            plt.show()


if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    linear = LinearRegression("Dataset/student-mat.csv", False)
    linear.save_load_model()
    linear.data_visualize(True)
