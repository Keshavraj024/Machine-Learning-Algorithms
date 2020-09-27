"""
Program to train the classifier user Support Vector Machine
"""
# Import Necessary Modules
import sklearn
from sklearn import datasets,model_selection,svm,metrics
import pickle

class svm_classifier:
	def __init__(self,save=None):
		self.data = datasets.load_breast_cancer()
		self.X = self.data.data
		self.Y = self.data.target
	
	# Split the data into train and test
	def train_test_splitter(self):
		self.x_train,self.x_test,self.y_train,self.y_test = \
			model_selection.train_test_split(self.X,self.Y,test_size=0.1)

		return self.x_train,self.x_test,self.y_train,self.y_test

	# Train the model
	def train(self):
		a,b,c,d = self.train_test_splitter()
		self.classifier = svm.SVC(kernel='poly')
		self.classifier.fit(a,c)
		return self.classifier

	# Calculate the accuracy
	def score(self):
		a,b,c,d = self.train_test_splitter()
		y_pred = self.train().predict(b)
		acc = metrics.accuracy_score(y_pred,d)
		return acc

	# Display the results
	def disp(self):
		print(f"Accuracy : {self.score()}")

	# Save or load the model
	def save_model(self):
		with open('svmmodel.pickle','wb') as file:
			pickle.dump(self.train(),file)
		print("Model saved to disk")

if __name__ == "__main__":
	classifier = svm_classifier()
	classifier.disp()
	classifier.save_model()
	