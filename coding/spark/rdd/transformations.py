# commonly used RDD transformations
import pyspark as ps
conf = ps.SparkConf().setMaster('local[4]').setAppName('My App')
sc = ps.SparkContext(conf=conf)
file_rdd = sc.textFile('../sample_data_files/cookie_data.txt')

# map
import json
data = file_rdd.map(json.loads).map(lambda x: (str(x.keys()[0]), int(x.values()[0])))

lines = sc.parallelize(['hello world', 'hi'])
print lines.collect()
words = lines.map(lambda line: line.split())
print words.collect()

# flatmap
lines = sc.parallelize(['hello world', 'hi'])
print lines.collect()
words = lines.flatMap(lambda line: line.split())
print words.collect()

# filter
jane_rdd = file_rdd.filter(lambda x: 'Jane' in x)


# ####set operations
rdd1 = sc.parallelize(['coffee', 'coffee', 'panda', 'monkey', 'tea'])
rdd2 = sc.parallelize(['coffee', 'money', 'kitty'])
# distinct
print rdd1.distinct().collect()
# union
print rdd1.union(rdd2).collect()
# intersection
print rdd1.intersection(rdd2).collect()
# subtract
print rdd1.subtract(rdd2).collect()
# cartesian
print rdd1.cartesian(rdd2).collect()





