import pyspark as ps

# in the command line, the command is python create_spark_context.py
# in the spark shell, i don't need to initialize a SparkContext since it is done automatically
conf = ps.SparkConf().setMaster('local[4]').setAppName('My App')
sc = ps.SparkContext(conf=conf)

