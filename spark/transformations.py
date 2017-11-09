# commonly used RDD transformations

# map


# filter
jane_rdd = file_rdd.filter(lambda x: 'Jane' in x)

# union
jane_rdd = file_rdd.filter(lambda x: 'Jane' in x)
duncan_rdd = file_rdd.filter(lambda x: 'Duncan' in x)
jane_duncan_rdd = jane_rdd.union(duncan_rdd)
