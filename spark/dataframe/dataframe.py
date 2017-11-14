#!/usr/bin/env python
import sys
import json
import joblib
import numpy as np
import pandas as pd
from time import time
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
df.show()
