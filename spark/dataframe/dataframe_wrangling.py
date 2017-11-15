import sys
import json
import joblib
import numpy as np
import pandas as pd
from time import time
from pyspark.sql import functions as F
from pyspark.sql import SQLContext, Row
from pyspark import SparkConf, SparkContext

path = '/Users/travis.howe/Projects/github/data_science/sample_data_files/X_train.txt'

conf = (SparkConf()
        .setAppName("data_frame_random_lookup")
        .set("spark.executor.instances", "10")
        .set("spark.executor.cores", 2)
        .set("spark.dynamicAllocation.enabled", "false")
        .set("spark.shuffle.service.enabled", "false")
        .set("spark.executor.memory", "500MB"))
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

t0 = time()

# read data as rdd
rdd_lines = sc.textFile(path).\
    map(lambda x: x.split()).\
    map(lambda p: Row(one=float(p[0]), two=float(p[1]), three=float(p[2])))

# create data frame
df = sqlContext.createDataFrame(rdd_lines)
# print df.show()
################ end of preliminaries

# reference a specific column
df.one
df['one']

# create a new column
df.withColumn('four', F.lit(np.random.rand()))

# manipulate column
df.withColumn('four', df['three'] * 2)
df.withColumn('four', df['three'] > 0)

# subset columns, create new columns
df.select('one', 'two')
df.select('one', (df['three'] > 0).alias('positive'))

# subset observations...i.e., masking
df.filter(df['three'] > 0).show()

# see also https://s3.amazonaws.com/assets.datacamp.com/blog_assets/PySpark_SQL_Cheat_Sheet_Python.pdf