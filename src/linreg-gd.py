# Shashank Gupta
# linreg-gd.py
#
# Standalone Python/Spark program to perform linear regression.
# Performs linear regression by computing the summation form of the
# closed form expression for the ordinary least squares estimate of beta.
# 
# TODO: Write this.
# 
# Takes the yx file as input, where on each line y is the first element 
# and the remaining elements constitute the x.
#
# Usage: spark-submit linreg.py <inputdatafile>
# Example usage: spark-submit linreg-gd.py <datafile> <alpha> <# of iterations>
#
#

import sys
import numpy as np

from pyspark import SparkContext

  
def getXList(line):
  # Deleting the first element y coordinate
  del line[0]
  # Inserting 1 for intercept beta-naught
  line.insert(0,1)
  m=""
  # Adding number to string matrix m
  for number in line:
    m=m+str(number)+";"
  # Removing the last semicolon
  m = m[:-1]
  # Converting to matrix float type format
  return np.matrix(m).astype('float').transpose()

# Function to calculate Xtranspose * (Y-XBeta)
def gd(y, x, beta):
  matBeta = np.asmatrix(beta)
  matY = np.asmatrix(y).astype('float')
  # Calculate XBeta
  xB = x * matBeta
  # Calculate (Y-XBeta)
  subtraction = matY - xB
  xTran = x.transpose()
  # Calculate Xtranspose * (Y-XBeta)
  secondTerm = xTran * subtraction
  return ('keyA', secondTerm)

  
if __name__ == "__main__":
  if len(sys.argv) !=4:
    print >> sys.stderr, "Usage: linreg-gd <datafile>"
    exit(-1)

  sc = SparkContext(appName="LinearRegressionGradientDescent")

  # Input yx file has y_i as the first element of each line 
  # and the remaining elements constitute x_i
  yxinputFile = sc.textFile(sys.argv[1])

  rdd = yxinputFile.map(lambda line: line.split(','))
  yxfirstline = rdd.first()
  yxlength = len(yxfirstline)
  #print "yxlength: ", yxlength

  # dummy floating point array for beta to illustrate desired output format
  beta = np.zeros((yxlength, 1), dtype=float)

  # Gradient descent alpha and iterations
  alpha = float(sys.argv[2])
  iterations = int(sys.argv[3])

  # Running for given number of iterations
  while iterations > 0:
    # Calculating Beta
    rddSecondTerm = rdd.map(lambda line: gd(line[0],getXList(line), beta))
	# Summation
    rddResult = rddSecondTerm.reduceByKey(lambda x,y : x+y)
	# Getting the value and multiplying by alpha
    result = rddResult.values().first()
    secondTermFinal = alpha * result
	# Getting the new beta
    beta = beta + secondTermFinal
    iterations = iterations - 1
  


  # Print the linear regression coefficients in desired output format
  print "beta: "
  for coeff in beta:
    print coeff

#  output_file = open('answer.out', 'w')
#  beta.tofile(output_file)
#  output_file.close()

  sc.stop()
