#!/usr/bin/env python
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from time import time
from pyspark.sql import functions as F
from pyspark.sql import SQLContext, Row
from pyspark import SparkConf, SparkContext


conf = (SparkConf()
        .setAppName("data_frame_random_lookup")
        .set("spark.executor.instances", "10")
        .set("spark.executor.cores", 2)
        .set("spark.dynamicAllocation.enabled", "false")
        .set("spark.shuffle.service.enabled", "false")
        .set("spark.executor.memory", "500MB")
        .set("spark.driver.extraClassPath", '/Users/travis.howe/Documents/RedshiftJDBC42-1.2.8.1005.jar')
        .set("spark.executor.extraClassPath", '/Users/travis.howe/Documents/RedshiftJDBC42-1.2.8.1005.jar')
        )

sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

df = sqlContext.read.format('jdbc').options(
    url=os.getenv('REDSHIFT_URL'),
    user=os.getenv('REDSHIFT_USER'),
    password=os.getenv('REDSHIFT_PASSWORD'),
    dbtable=os.getenv('REDSHIFT_DB')
).load()
df.show()

