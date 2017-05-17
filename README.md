# LinearRegressionModel
Implementing Linear Regression model using Spark

src : Consists of python source code linreg.py linreg-gd.py 

out : Consists of the two output files yxlin.out yxlin2.out using ordinary least squares estimate

gd-out : Consists of the two output files yxlin.out yxlin2.out using gradient descent

#########################################################################################

Used the closed form expression for the ordinary least squares estimate of the linear regression coefficients

Steps for running the code:

-> Putting the input files in the hdfs  
hdfs dfs -put yxlin.csv
hdfs dfs -put yxlin2.csv

-> Going to the root directory and then going to the bin directory  
cd /usr/lib/spark/bin

-> Running the spark program for the two input files   
spark-submit /users/shashank/linreg.py yxlin.csv
spark-submit /users/shashank/linreg.py yxlin2.csv


########################################################################################

Gradient Descent approach to calculate regression coefficients : alpha= 0.001 and iterations = 500

The following formula is used for gradient descent:
beta = beta+alpha*summation(Xtrans (Y - X*beta))

First, the summation term is calculated by mapper and emitted as "keyA". The reducer adds up all the terms which has "keyA".
The above term is multiplied by alpha and added to previous beta value to get the new beta value.

I have assumed the initial Beta value to be a matrix of all 0's.

Steps for running the code:

-> Putting the input files in the hdfs  
hdfs dfs -put yxlin.csv
hdfs dfs -put yxlin2.csv

-> Going to the root directory and then going to the bin directory  
cd /usr/lib/spark/bin

-> Running the spark program for the two input files  
spark-submit /users/shashank/linreg-gd.py yxlin.csv 0.001 500
spark-submit /users/shashank/linreg-gd.py yxlin2.csv 0.001 500

