#!/usr/bin/env python
import os
import sys
import json
import joblib
import datetime
import numpy as np
import pandas as pd
from time import time

from pyspark.sql import functions as F
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row, types
from pyspark.ml.regression import LinearRegression


conf = (SparkConf()
        .setAppName("data_frame_random_lookup")
        .set("spark.driver.host", "localhost")
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

query = '''
(select
    databunker.applications.*,
    databunker.agents.agent_name, databunker.agents.start_date as agent_start_date, databunker.agents.term_date as agent_term_date, databunker.agents.business, databunker.agents.resident_state_id,
    databunker.agent_annotations.start_date, databunker.agent_annotations.gross_policy_budget
from databunker.applications
left join databunker.agents on databunker.agents.id = databunker.applications.agent_id
left join databunker.agent_annotations on databunker.agent_annotations.agent_id = databunker.agents.id
where date_part('month', databunker.applications.submission_date) = date_part('month', databunker.agent_annotations.start_date) AND
      date_part('year', databunker.applications.submission_date) = date_part('year', databunker.agent_annotations.start_date)) ali
'''

df = sqlContext.read.format('jdbc').options(
    url=os.getenv('REDSHIFT_URL'),
    user=os.getenv('REDSHIFT_USER'),
    password=os.getenv('REDSHIFT_PASSWORD'),
    # dbtable=os.getenv('REDSHIFT_DB')
    dbtable=query
).load()

df = df.withColumn('tenure', F.datediff(df['agent_term_date'], df['agent_start_date']))
df = df.withColumn('tenure', F.when(df['tenure'] < 0, None).otherwise(df['tenure']))
df = df.withColumn('start_month', F.month(df['agent_start_date']))

# #drop nulls
# #these do the same thing
# df = df.filter(df['start_month'].isNotNull())
df = df.where(F.col('start_month').isNotNull())

df = df.select('tenure', 'gross_policy_budget', 'start_month')
dummylist = ['start_month']

# #create dummy variables
def create_dummies(df, dummylist):
    for inputcol in dummylist:
        categories = df.select(inputcol).rdd.distinct().flatMap(lambda x: x).collect()

        exprs = [F.when(F.col(inputcol) == category, 1).otherwise(0).alias(category) for category in categories]

        for index, column in enumerate(exprs):
            df = df.withColumn(inputcol + str(index), column)

    return df

df = create_dummies(df, dummylist)



df.show()



# todo: create dummies
# todo: drop nulls
# todo: is there like a pipe in pyspark?



sys.exit()
train_data, test_data = df.randomSplit([.8, .2], seed=22)
train_data.show()


lr = LinearRegression(labelCol='label', maxIter=10, regParam=0.3, elasticNetParam=0.8)

linearModel = lr.fit(train_data)
predicted = linearModel.transform(test_data)
labels = predicted.select('label').df.map(lambda x: x[0])


sc.stop()



# https://www.datacamp.com/community/tutorials/apache-spark-tutorial-machine-learning