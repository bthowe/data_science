# from text file to rdd
file_rdd = sc.textFile('data/cookie_data.txt')

# todo: from csv file to rdd
file_rdd = sc.textFile("data/file.csv").map(lambda line: line.split(","))  # still need to format columns

# something from memory...not widely used since requires the entire dataset in memory on one machine
file_rdd2 = sc.parallelize(np.random.uniform(0, 1, size=(20, 5)).tolist())
