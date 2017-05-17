# Shashank Gupta
# linreg.py
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
# Example usage: spark-submit linreg.py yxlin.csv
#
#

import sys
import numpy as np

from pyspark import SparkContext


  
def matrixMultiplicationXX(matX):
  # X Transpose
  matXTranspose = matX.transpose()
  # X * X Transpose
  result = matX * matXTranspose
  return ('keyA', result)

  
def matrixMultiplicationXY(y,x):
  matY = np.asmatrix(y).astype('float')
  # X * Y
  result = x * matY
  return ('keyB', result)
  
  
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
  return np.matrix(m).astype('float')
  

  
if __name__ == "__main__":
  if len(sys.argv) !=2:
    print >> sys.stderr, "Usage: linreg <datafile>"
    exit(-1)

  sc = SparkContext(appName="LinearRegression")

  # Input yx file has y_i as the first element of each line 
  # and the remaining elements constitute x_i
  yxinputFile = sc.textFile(sys.argv[1])

  rdd = yxinputFile.map(lambda line: line.split(','))
  yxfirstline = rdd.first()
  yxlength = len(yxfirstline)
  #print "yxlength: ", yxlength

  # dummy floating point array for beta to illustrate desired output format
  beta = np.zeros(yxlength, dtype=float)

  # Getting the X list
  rddXX = rdd.map(lambda line:getXList(line))
  rddXXResult = rddXX.map(matrixMultiplicationXX)
  
  # Calculating summation of X * X transpose
  rddFirstTerm = rddXXResult.reduceByKey(lambda x,y : x+y)
  rddFirstTerm.collect()
  
  # Calculating the second term 
  rddXY = rdd.map(lambda line: matrixMultiplicationXY(line[0],getXList(line)))
  rddSecondTerm = rddXY.reduceByKey(lambda x,y : x+y)
  rddSecondTerm.collect()
  
  # Retrieving the first matrix value
  matXX = rddFirstTerm.values().first()
  
  # Getting the inverse
  matXXInv = np.asmatrix(np.linalg.inv(matXX))
  
  # Retrieving the second matrix value
  matXY = rddSecondTerm.values().first()
  
  # Multiplying first term with second 
  matResult = matXXInv * matXY
  
  # Final value
  beta = np.squeeze(np.asarray(matResult))



  # Print the linear regression coefficients in desired output format
  print "beta: "
  for coeff in beta:
    print coeff

#  output_file = open('answer.out', 'w')
#  beta.tofile(output_file)
#  output_file.close()

  sc.stop()
