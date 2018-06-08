import numpy as np

def loadData(dataFile):
		tempArray = np.loadtxt(dataFile, dtype=float, delimiter=",")
		lenX,lenY = np.shape(tempArray)
		y = np.asmatrix(tempArray[:,lenY-1])
		y = np.transpose(y)
		x = tempArray[:, :lenY -1]
		return x,y


x,y = loadData("test.txt")
print(x)
newMatrix = np.transpose(np.asmatrix(x[:,1]))
for i in range(2):
	newMatrix = np.multiply(newMatrix, newMatrix)

newMatrix = np.asmatrix(newMatrix)

x = np.concatenate((newMatrix, x), axis=1)
x = np.concatenate((x, y), axis=1)
np.savetxt("test.txt", x, delimiter=",")