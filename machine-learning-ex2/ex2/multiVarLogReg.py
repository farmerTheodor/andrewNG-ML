import numpy as np
import matplotlib.pyplot as plt

class LogReg(object):
	"""docstring for LogReg"""
	def __init__(self, docName):
		super(LogReg, self).__init__()
		theta = None
		x = None
		y = None
		m = 0
		numFeatures = 0
		self.loadData(docName)
		self.addOnes()


	def toThePowerOf(self,feature, powerOf):
		newMatrix = np.transpose(np.asmatrix(self.x[:,feature]))
		print(np.shape(self.x))
		print(np.shape(newMatrix))
		for x in range(powerOf):
			newMatrix = np.multiply(self.x[:,feature],self.x[:,feature])
		
		newMatrix = np.asmatrix(newMatrix)
		print(np.shape(newMatrix))
		self.x = np.asmatrix(np.concatenate((newMatrix, self.x), axis=1))
		
		self.numFeatures = self.numFeatures + 1

	def addOnes(self):
		ones = np.asmatrix(np.ones((self.m,1)))
		self.x = np.asmatrix(np.concatenate((ones, self.x), axis=1))
		self.m,self.numFeatures = np.shape(self.x)
	

	def initTheta(self, thetaInit):
		if(np.size(thetaInit) < self.numFeatures):
			print("invalid numTheta")
			return 1
		else:
			self.theta = np.asmatrix(thetaInit)
			return 0

	def loadData(self, dataFile):
		tempArray = np.loadtxt(dataFile, dtype=float, delimiter=",")
		lenX,lenY = np.shape(tempArray)
		self.y = np.asmatrix(tempArray[:,lenY-1])
		self.y = np.transpose(self.y)
		self.x = tempArray[:, :lenY -1]
		self.m,self.numFeatures = np.shape(self.x)

	def normalizeData(self):
		average = self.x.sum(axis=0)/self.m
		rangeOf = self.x.max(axis=0) -  self.x.min(axis=0)
		self.x = (self.x - average)/rangeOf
		"""
		hypothesis = sigmoid(X*theta);
		diff = y.*(-log(hypothesis)) + (1-y).*(-log(1-hypothesis));
		excludeTheta0 = theta(2:end);
		J = (sum(diff)/m)+ (lambda/(2*m) )*sum(excludeTheta0.^2);
		thetaProduct = (lambda/m)*excludeTheta0;
		thetaProduct = vertcat([0], thetaProduct)
		grad = sum((hypothesis - y)'*X,1)'/m + thetaProduct;
		print(grad)
		"""


	def calculateCost(self, curLambda):
		hypothesis = 1/(1+np.exp(-1*self.x*np.transpose(self.theta)))
		dif = np.multiply(self.y * -1, np.log(hypothesis)) - np.multiply((1-self.y),(np.log(1 - hypothesis)))
		sizeOfTheta = (curLambda/(2*self.m))*np.sum(np.multiply( self.theta[:,1:], self.theta[:,1:]))
		cost = np.sum(dif)/self.m + sizeOfTheta
		return cost

	def calculateNewTheta(self,curLambda, alpha):
		hypothesis = 1/(1+np.exp(-1*self.x*np.transpose(self.theta)))
		dif = hypothesis - self.y
		thetaProduct = (curLambda/self.m)*self.theta[:,1:]
		thetaProduct = np.insert(thetaProduct, 0, 0)

		grade = (np.sum(np.transpose(dif)*self.x,axis=0)/self.m )+ thetaProduct

		self.theta = self.theta - (alpha*grade)
		#zeroTheta = self.theta[:,0] - (alpha*(np.sum(grade[:,0], axis=0)/self.m))
		#newTheta = self.theta[:,1:] - alpha*(((np.sum(grade[:,1:], axis=0)/self.m) - (self.theta[:,1:]*(curLambda/self.m))))
		#self.theta = np.concatenate((zeroTheta, newTheta), axis=1)

	def plotData(self):
		positve1 = self.x[:,1]
		negative1 = positve1
		negative1 = negative1[self.y==0]
		positve1 = positve1[self.y == 1]
		positve2 = self.x[:,2]
		negative2 = positve2
		negative2 = negative2[self.y==0]
		positve2 = positve2[self.y == 1]
		plt.plot(positve1,positve2,"bs", negative1,negative2,"g^")
		plt.show()

def main():
	logML = LogReg("ex2data1.txt")
	if(logML.initTheta([0,0,0]) == 1):
		return 
	curLambda = 10
	alpha = .0003
	
	for x in range(1000000):
		#print(logML.theta)
		curCost = logML.calculateCost(curLambda)
		print(logML.theta)
		print("cost == ",curCost)
		logML.calculateNewTheta(curLambda, alpha)

if __name__ == '__main__':
	main()
