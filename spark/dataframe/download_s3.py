#!/usr/bin/env python
import os
from pyspark.sql import SQLContext
from pyspark import SparkConf, SparkContext

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

# #makes rdd
# text_file = sc.textFile('s3n://bucket/file.txt')
# print text_file.collect()

# #makes df
# text_file = sqlContext.read.load('s3n://bucket/file.csv', format='csv')
text_file = sqlContext.read.text('s3n://bucket/file.txt')
print text_file.show()
