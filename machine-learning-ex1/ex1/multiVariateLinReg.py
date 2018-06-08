import numpy as np



class LinReg(object):
	"""docstring for LinReg"""
	def __init__(self, dataFile):
		super(LinReg, self).__init__()
		theta = None
		x = None
		y = None
		#m = number of training samples
		m = 0
		numFeatures = 0
		self.loadData(dataFile)
		self.normalizeData()
		self.addOnes()


	def addOnes(self):
		ones = np.asmatrix(np.ones((self.m,1)))
		self.x = np.asmatrix(np.concatenate((ones, self.x), axis=1))
		self.m,self.numFeatures = np.shape(self.x)
		

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


	def initTheta(self, thetaInit):
		if(np.size(thetaInit) < self.numFeatures):
			return 1
		else:
			self.theta = np.asmatrix(thetaInit)
			return 0

	def computeCost(self):
		j = 0
		hypothesis = np.transpose(self.theta*np.transpose(self.x))
		error = hypothesis - self.y
		error = np.multiply(error, error)
		j = np.transpose(error).sum()
		j = j/(2*self.m)
		return j

	def computeNewTheta(self, learningRate):
		hypothesis = np.transpose(self.theta*np.transpose(self.x))
		error = hypothesis - self.y
		costFuncDeriv =  np.multiply(error,self.x)
		costFuncDeriv = costFuncDeriv.sum(axis=0)
		costFuncDeriv = costFuncDeriv / self.m
		newTheta = self.theta - learningRate*costFuncDeriv

		return newTheta


	def runAlgo(self, numIterations, learningRate):
		cost = []

		for x in range(numIterations):
#			print("this is the cost")

			cost.append(self.computeCost())
#			print(cost[x])
			self.theta = self.computeNewTheta(learningRate)
#			print(self.theta)

		return(cost[len(cost)-1])
		pass

def main():
	newLin = LinReg("ex1data2.txt")
	if(newLin.initTheta([0,0,0])):
		print("error: incorrect initTheta")
		return 
	currentCost = newLin.computeCost()
	print(currentCost)
	currentCost = newLin.runAlgo(4000, .1)
	print(currentCost)



if __name__ == '__main__':
	main()
