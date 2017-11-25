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
from pyspark.ml.linalg import DenseVector
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row, types
from pyspark.ml.feature import VectorAssembler
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
        .set("spark.jars.packages", 'org.apache.hadoop:hadoop-aws:2.7.1')
        )

sc = SparkContext(conf=conf)
AWS_KEY = os.getenv('AWS_KEY')
AWS_SECRET = os.getenv('AWS_SECRET')
sc._jsc.hadoopConfiguration().set('fs.s3n.awsAccessKeyId', AWS_KEY)
sc._jsc.hadoopConfiguration().set('fs.s3n.awsSecretAccessKey', AWS_SECRET)

sqlContext = SQLContext(sc)

def raw_create():
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
            dbtable=query
        ).load()

#     todo: df save in s3

def preprocess():
    df = sqlContext.read.text('s3n://bucket/Travis/file.txt')

    df = df.withColumn('tenure', F.datediff(df['agent_term_date'], df['agent_start_date']))
    df = df.withColumn('tenure', F.when(df['tenure'] < 0, None).otherwise(df['tenure']))
    df = df.withColumn('start_month', F.month(df['agent_start_date']))

    df = df.select('tenure', 'gross_policy_budget', 'start_month')
    df = df. \
        where(F.col('start_month').isNotNull()). \
        where(F.col('tenure').isNotNull()). \
        where(F.col('gross_policy_budget').isNotNull())

    def create_dummies(df, dummylist, drop=True):
        for inputcol in dummylist:
            categories = df.select(inputcol).rdd.distinct().flatMap(lambda x: x).collect()

            exprs = [F.when(F.col(inputcol) == category, 1).otherwise(0).alias(str(category)) for category in categories]

            for index, column in enumerate(exprs):
                df = df.withColumn(inputcol + str(index), column)

            if drop:
                df = df.drop(inputcol)

        return df

    dummylist = ['start_month']
    df = create_dummies(df, dummylist)

    input_data = df.rdd.map(lambda x: (x[0], DenseVector(x[1:])))
    df = sqlContext.createDataFrame(input_data, ['label', 'features'])
    return df


def train_test_create():
    df = sqlContext.read.text('s3n://bucket/Travis/file.txt')

    train_data, test_data = df.randomSplit([.8, .2], seed=22)

#     todo: save train and test in s3

def model_train():
    train_data = sqlContext.read.text('s3n://bucket/Travis/train_data.txt')

    # todo: validation and tuning
    lr = LinearRegression(labelCol='label', maxIter=10, regParam=0.3, elasticNetParam=0.8)
    linearModel = lr.fit(train_data)
    # todo: save model in s3

def model_evaluate():
    # todo: load model from s3
    test_data = sqlContext.read.text('s3n://bucket/Travis/test_data.txt')
    predicted = linearModel.transform(test_data)

    predictions = predicted.select("prediction").rdd.map(lambda x: x[0])
    labels = predicted.select('label').rdd.map(lambda x: x[0])
    predictionAndLabel = predictions.zip(labels)
    print predictionAndLabel.collect()

    # todo: scoring
    print linearModel.summary  # ?


if __name__ == '__main__':
    raw_create()
    preprocess()
    train_test_create()
    model_train()
    sc.stop()

# todo: pipeline and pipeline wrapper
