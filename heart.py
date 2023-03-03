import numpy as np
import matplotlib.pyplot as mtp  
import pandas as pd

def estimate_coef(x, y):
	n = np.size(x)  #since x and y are same size

	# mean of x and y vector
	m_x = np.mean(x)
	m_y = np.mean(y)

	# calculating cross-deviation and deviation about x
	SS_xy = np.sum(y*x) - n*m_y*m_x
	SS_xx = np.sum(x*x) - n*m_x*m_x

	# calculating regression coefficients
	b_1 = SS_xy / SS_xx
	b_0 = m_y - b_1*m_x

	return (b_0, b_1)
    
def plot_regression_line(x, y, b):
	# plotting the actual points as scatter plot
	plt.scatter(x, y, color = "m",
			marker = "o", s = 30)

	# predicted response vector
	y_pred = b[0] + b[1]*x

	# plotting the regression line
	plt.plot(x, y_pred, color = "g")

	# putting labels
	plt.xlabel('x')
	plt.ylabel('y')

	# function to show plot
	plt.show()
  
#importing datasets  
data_set= pd.read_csv('heart.csv')  

#Extracting Independent and dependent Variable  
data_set.DEATH_EVENT.value_counts()

X=data_set.drop(["DEATH_EVENT"],axis=1)
y=data_set["DEATH_EVENT"]
Y=data_set["DEATH_EVENT"].tolist()
print(Y)

 # Splitting the dataset into training and test set.

from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(X, y, test_size= 0.25, random_state=0)

from sklearn.tree import DecisionTreeClassifier  
classifier= DecisionTreeClassifier(criterion='entropy', random_state=0)  
classifier.fit(x_train, y_train)

y_pred= classifier.predict(x_test)

print(y_pred)

def main():
	# observations / data
	x =x_test.reshape(75,12)
	y =y_pred.reshape(75,12)

	# estimating coefficients
	b = estimate_coef(x, y)
	print("Estimated coefficients:\nb_0 = {} \
		\nb_1 = {}".format(b[0], b[1]))

	# plotting regression line
	plot_regression_line(x, y, b)

if __name__ == "__main__":
	main()


#from sklearn.metrics import accuracy_score
#acc=accuracy_score(y_test, y_pred)
#print(acc*100)


