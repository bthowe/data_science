# from text file to rdd
file_rdd = sc.textFile('data/cookie_data.txt')

# todo: from csv file to rdd
file_rdd = sc.textFile("data/file.csv").map(lambda line: line.split(","))  # still need to format columns

