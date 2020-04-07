"""
Simple script that creates nonlinear regression model,
made in around 1h to help my friend at college.
Function that I found, which fits given data set, was f(x)= a*(x-b)**2 + c
so the point was to find the best values for parameters a, b, c.
"""

from random import shuffle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import warnings


alldata = []
datax = []
datay = []

with open('dane9.txt', 'r') as file:
    for line in file:
        rowdata = line.split(" ")
        alldata.append([float(rowdata[0]), float(rowdata[1])])


shuffle(alldata)
splitindex = round(len(alldata)*0.7)
trainset = alldata[:splitindex]
testset = alldata[splitindex:]

trainx, trainy = zip(*trainset)

testx, testy = zip(*testset)

plt.scatter(trainx, trainy)
plt.scatter(testx, testy, c='red')
# plt.savefig('alldata.png')
plt.show()

x = np.array(trainx).reshape((-1, 1))
y = np.array(trainy)

model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
b0 = model.intercept_
b1 = model.coef_
xtest = np.array(testx).reshape((-1, 1))
predy = model.predict(xtest)
rsum = 0

for test_y, pred_y in zip(testy, predy):
    rsum += (test_y - pred_y)**2

plt.scatter(testx, testy, c='black')
plt.scatter(testx, list(predy), c='green')
plt.show()
#plt.savefig('linearpredict.png')

print('Sumofsquarederror of predicted data based on testing set with linear function: ', rsum)

def functionadv(xvals, bottomlimit, startval, stretch):
    yvals = []
    for x in xvals:
        yvals.append(stretch*(x-startval)**2 + bottomlimit)
    return yvals


def generateparameters(x_vals, y_vals):

    plot = False
    if plot:
        plt.scatter(x_vals, y_vals)
        plt.show()

    def sumOfSquaredError(parameterTuple):
        warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm


        val = functionadv(x_vals, *parameterTuple)


        sum = 0
        for i in range(len(x_vals)):
            sum += (y_vals[i] - val[i]) ** 2
        return sum


    def generate_Initial_Parameters():
        # min and max used for bounds
        maxX = max(x_vals)
        minX = min(x_vals)
        maxY = max(y_vals)
        minY = min(y_vals)

        parameterBounds = []


        parameterBounds.append([0, 2]) # seach bounds for bottomlimit
        parameterBounds.append([1.2, 2.5]) # seach bounds for startval
        parameterBounds.append([1, 2]) # seach bounds for stretch

        # "seed" the numpy random number generator for repeatable results
        result = differential_evolution(sumOfSquaredError, parameterBounds, seed=3)
        return result.x

    # generate initial parameter values
    geneticParameters = generate_Initial_Parameters()

    # curve fit the test data
    fittedParameters, pcov = curve_fit(functionadv, x_vals, y_vals, geneticParameters)
    return fittedParameters

fittedparams = generateparameters(trainx, trainy)

predictedvals = functionadv(testx, *fittedparams)

plt.scatter(testx, testy, c='black')
plt.scatter(testx, predictedvals, c='green')
plt.show()
# plt.savefig('squarepredict.png')

rsum = 0
for test_y, pred_y in zip(testy, predictedvals):
    rsum += (test_y - pred_y)**2
print('Sumofsquarederror of predicted data based on testing set with ^2 function: ', rsum)
