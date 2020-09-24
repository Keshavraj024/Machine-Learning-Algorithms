"""
Program to implement K-Nearest Neighbor to a classification problem
"""
# Import the necessary module
import pandas as pd
from sklearn import model_selection,preprocessing
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pickle

class knn:
	# Initialisation
	def __init__(self,data):
		self.data = pd.read_csv(data)
		self.df = pd.DataFrame()
		
	# Proprocess the data
	def data_preprocessing(self):
		encode = preprocessing.LabelEncoder()
		for i in self.data.columns:
			a = list(self.data[i])
			if a[0].isalpha() or a[0].isnumeric():
				self.df[i] = encode.fit_transform(a)
			else:
				self.df[i] = self.data[i]
		return self.df
	
	# Retrieve the train and test dataset
	def train_test_data(self):
		self.x = np.array(self.data_preprocessing()[['buying','maint','doors','persons','lug_boot',
							'safety']])
		self.y = np.array(self.data_preprocessing()[['class']])
		return self.x,self.y
		
	# Split the data into test and train set
	def train_test_split(self):
		self.x_train,self.x_test,self.y_train,self.y_test = \
			model_selection.train_test_split(self.train_test_data()[0]
					,self.train_test_data()[1],test_size=0.1)
		return self.x_train,self.x_test,self.y_train,self.y_test
	
	# Train the model
	def train_model(self):
		a,b,c,d = self.train_test_split()
		self.knn_classifier = KNeighborsClassifier(n_neighbors=9)
		self.knn_classifier.fit(a,c)
		return self.knn_classifier

	# Get the score prodicted by the model
	def accuracy(self):
		accuracy = self.train_model().score(self.x_test,self.y_test)
		return accuracy

	# Display the results
	def run(self):
		print(f"Accuracy : {self.accuracy()}")


	# Save the model
	def save_model(self):
		with open("Model/carmodel.pickle","wb") as file:
			pickle.dump(self.knn_classifier,file)
		print("Model save to disk")

if __name__ == "__main__":
	classifier = knn('Dataset/car.data')
	classifier.run()
	classifier.save_model()