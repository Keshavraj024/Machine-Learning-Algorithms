"""
Program to train the Linear model
"""
# Import the necessary modules
import pandas as pd
from sklearn import model_selection,linear_model
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import style

class machine_learning:
	# Initialize the  data
	def __init__(self,data,train):
		self.data = pd.read_csv(data,sep=";")
		self.train = train
	
	# Split the data into train an test set
	def train_test_split(self):
		self.x = np.array(self.data[['G1','G2','absences','failures','studytime']])
		self.y = np.array(self.data[['G3']])
		self.x_train,self.x_test,self.y_train,self.y_test = \
			model_selection.train_test_split(self.x,self.y,test_size= 0.1)
		return self.x_train,self.x_test,self.y_train,self.y_test
	
	# Train the linear model
	def fit(self):
		self.a,self.b,self.c,self.d = self.train_test_split()
		self.regressor =  linear_model.LinearRegression()
		self.regressor.fit(self.a,self.c)
		return self.regressor

	# Save the model
	def save_load_model(self):
		if self.train == True:
			self.regressor = self.fit()
			with open("Model/stdmodel.pickle","wb") as file:
				pickle.dump(self.regressor,file)
			print("Model is saved")
		else:
			with open("Model/stdmodel.pickle","rb") as file:
				self.model = pickle.load(file)
			print("Model is loaded")
			self.disp()
			
	# Get the accuracy of the model
	def score(self):
		self.a,self.b,self.c,self.d = self.train_test_split()
		accuracy = self.model.score(self.b,self.d)
		return accuracy

	# Get the model parameters
	def model_parameters(self):
		coefficients = self.model.coef_
		intercepts = self.model.intercept_
		return coefficients,intercepts

	# Print Everything to Screen
	def disp(self):
		print(f"Accuracy     :  {self.score()}")
		print(f"coefficients : {self.model_parameters()[0]}")
		print(f"Intercepts   :  {self.model_parameters()[1]}")

	# Data Visualization
	def data_visualize(self,visualize):
		if visualize == True:
			style.use('ggplot')
			plt.scatter(self.data[['G1']],self.data[['G2']])
			plt.xlabel('G1')
			plt.ylabel('G3')
			plt.show()

if __name__ == "__main__":
	linear = machine_learning('Dataset/student-mat.csv',False)
	linear.save_load_model()
	linear.data_visualize(True)
