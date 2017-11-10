import sys
import json
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer

import pyspark as ps
from pyspark.ml.linalg import Vectors
from pyspark.sql import DataFrameReader as spark
from pyspark.sql import functions as F
from pyspark.sql.types import *




import os
import numpy as np

import pyspark as ps

dlds = '/Users/travis.howe/Downloads/'

# to run this in the command line, simply python galvanize_individual.py
# in the spark shell, i don't need to initialize SparkContext (done automatically)
conf = ps.SparkConf().setMaster('local[4]').setAppName('My App')
sc = ps.SparkContext(conf=conf)


# file_rdd = sc.textFile('../sample_data_files/cookie_data.txt')
rdd1 = sc.parallelize(range(10))
print rdd1.reduce(lambda x, y: x + y)



# start with page 67, aggregate


sys.exit()

import json
data = file_rdd.map(json.loads).map(lambda x: (str(x.keys()[0]), int(x.values()[0])))

print data.filter(lambda x: x[1] > 5).collect()

print data.reduceByKey(lambda x, y: max(x, y)).collect()

print data.values().reduce(lambda x, y: x + y)





# .persist  # ...persists the rdd, which would be useful if I'm doing a few different computations using that particular dataset
#










# used to use pyspark.sql.SQLContext(), but has been replaced by the following
spark = ps.sql.SparkSession.builder.master("local").appName("Word Count").getOrCreate()
# spark = ps.sql.SparkSession.builder.master("local").appName("Word Count").config("spark.some.config.option", "some-value").getOrCreate()

df = spark.read.text('../sample_data_files/cookie_data.txt')  #.selectExpr("value as json")
df.show()
sys.exit()


print [f.dataType for f in df.schema.fields]


# df = spark.read.json('galvanize_spark/data/cookie_data.txt')  # results in the keys being the column names
# df.show()


# df.withColumn('C', F.lit(0))
# print df.show()

import ast
map_udf_key = F.udf(lambda x: ast.literal_eval(x).keys()[0], returnType=StringType())
map_udf_val = F.udf(lambda x: ast.literal_eval(x).values()[0], returnType=IntegerType())
df.withColumn('key', map_udf_key('json')).withColumn('value', map_udf_val('json')).show()




# print df.collect()

sys.exit()


# todo: or I could try building out the pipeline magic from before
