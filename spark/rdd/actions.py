# commonly used actions
# can often iterate over the object returned
import pyspark as ps
dlds = '/Users/travis.howe/Downloads/'
conf = ps.SparkConf().setMaster('local[4]').setAppName('My App')
sc = ps.SparkContext(conf=conf)
file_rdd = sc.textFile('../sample_data_files/cookie_data.txt')

# reduce
rdd1 = sc.parallelize(range(10))
print rdd1.reduce(lambda x, y: x + y)

# will print the entire rdd to screen
print file_rdd.collect()

# will print the first row of rdd to screen
print file_rdd.first()

# will print the first twenty rows of the rdd to screen
print file_rdd.take(20)

# will save the rdd as a text file
file_rdd.saveAsTextFile(dlds + 'thing.txt')



