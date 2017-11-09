# commonly used actions
# can often iterate over the object returned

# will print the entire rdd to screen
print file_rdd.collect()

# will print the first row of rdd to screen
print file_rdd.first()

# will print the first twenty rows of the rdd to screen
print file_rdd.take(20)

# will save the rdd as a text file
saveAsTextFile()


