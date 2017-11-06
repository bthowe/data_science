import os
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


# used to use pyspark.sql.SQLContext(), but has been replaced by the following
spark = ps.sql.SparkSession.builder.master("local").appName("Word Count").getOrCreate()
# spark = ps.sql.SparkSession.builder.master("local").appName("Word Count").config("spark.some.config.option", "some-value").getOrCreate()

df = spark.read.text('galvanize_spark/data/cookie_data.txt').selectExpr("value as json")
df.show()


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
