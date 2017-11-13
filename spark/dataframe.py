#!/usr/bin/env python
import sys
import json
import joblib
import numpy as np
import pandas as pd
from time import time
from pyspark.sql import *
from pyspark import SparkConf, SparkContext

# print pd.read_table('/Users/travis.howe/Projects/github/data_science/sample_data_files/cookie_data.txt')
# df = pd.DataFrame(np.random.uniform(-1, 1, size=(20, 3)))
# np.savetxt('/Users/travis.howe/Projects/github/data_science/sample_data_files/X_train.txt', df.values)
# sys.exit()


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


path = '/Users/travis.howe/Projects/github/data_science/sample_data_files/X_train.txt'
lines = sc.textFile(path)  #.map(float)
# lines = sc.textFile(path).map(lambda x: float(x))
# lines = sc.textFile(path).map(lambda p: Row(one=float(p[0]), two=float(p[1]), three=float(p[2])))
print lines.collect()
sys.exit()

## create data frame
orders_df = sqlContext.createDataFrame(lines.map(lambda l: l.split(",")))

        # .map(lambda p: Row(cust_id=int(p[0]), order_id=int(p[1]), email_hash=int(p[2]), ssn_hash=int(p[3]), product_id=int(p[4]))))
print orders_df.show()
sys.exit()

## filter where the order_id, the second field, is equal to 96922894
orders_df.where(orders_df['order_id'] == 96922894).show()

tt = str(time() - t0)
print "DataFrame performed in " + tt + " seconds"