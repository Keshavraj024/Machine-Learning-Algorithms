#!usr/bin/python3
"""
Program to implement K-Nearest Neighbor to a classification problem
"""
import pandas as pd
from sklearn import model_selection,preprocessing
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pickle
import logging

class Knn:
	def __init__(self,data)->None:
		self.data = pd.read_csv(data)
		
	def data_preprocessing(self)->pd.DataFrame:
		"""Proprocess the data

		Returns:
			_type_: DataFrame
		"""
		df = pd.DataFrame()
		encode = preprocessing.LabelEncoder()
		for col in self.data.columns:
			a = list(self.data[col])
			if a[0].isalpha() or a[0].isnumeric():
				df[col] = encode.fit_transform(a)
			else:
				df[col] = self.data[col]
		return df
	
	def train_test_data(self)->tuple:
		"""Generate train and test data

		Returns:
			_type_: numpy array
		"""
		self.x = np.array(self.data_preprocessing()[['buying','maint','doors','persons','lug_boot',
							'safety']])
		self.y = np.array(self.data_preprocessing()[['class']])
		return self.x,self.y
		
	def train_test_split(self)->tuple:
		"""Split the data into test and train set

		Returns:
			_type_: array
		"""
		self.x_train,self.x_test,self.y_train,self.y_test = \
			model_selection.train_test_split(self.train_test_data()[0]
					,self.train_test_data()[1],test_size=0.1)
		return self.x_train,self.x_test,self.y_train,self.y_test
	
	def train_model(self)->KNeighborsClassifier:
		"""Train the model

		Returns:
			_type_: KNNClassifier
		"""
		train_x,_,train_y,_ = self.train_test_split()
		self.knn_classifier = KNeighborsClassifier(n_neighbors=9)
		self.knn_classifier.fit(train_x,train_y)
		return self.knn_classifier

	def accuracy(self)->float:
		"""Calculate the accuracy

		Returns:
			_type_: number
		"""
		accuracy = self.train_model().score(self.x_test,self.y_test)
		return accuracy

	def run(self):
		"""Display the results
		"""
		logging.info(f"Accuracy : {self.accuracy()}")


	def save_model(self):
		"""Save the model
		"""
		with open("Model/carmodel.pickle","wb") as file:
			pickle.dump(self.knn_classifier,file)
		logging.info("Model saved to disk")

if __name__ == "__main__":
	logger = logging.getLogger()
	logger.setLevel(logging.INFO)
	classifier = Knn('Dataset/car.data')
	classifier.run()
	classifier.save_model()