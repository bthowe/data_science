#!/usr/bin/env python
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

### groupBy
gb = df.groupBy(df['one'])

# count()
results = gb.count()  #.sort("count",ascending=False)
# print results.show()

# agg()
results = gb.agg({'*': 'count'})
gb.agg(F.avg('one'), F.min('two'), F.max('three')).show()  # use .alias to rename, e.g. F.avg('one').alias('yo')

# also, mean(), max(), min(), sum()

# pivot...leaves indeces unchanged and makes the columns equal to the values in the specified column name
results = gb.pivot('one').count()
print results.show()


