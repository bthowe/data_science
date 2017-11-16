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

conf = (SparkConf()
        .setAppName("data_frame_random_lookup")
        .set("spark.executor.instances", "10")
        .set("spark.executor.cores", 2)
        .set("spark.dynamicAllocation.enabled", "false")
        .set("spark.shuffle.service.enabled", "false")
        .set("spark.executor.memory", "500MB"))
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

df = sqlContext.load(source="jdbc",
                     url="jdbc:postgresql://host:port/dbserver?user=yourusername&password=secret",
                     dbtable="schema.table")


# todo: jar file, url, query
# https://community.hortonworks.com/articles/59205/spark-pyspark-to-extract-from-sql-server.html
# maybe put jar file in this directory
