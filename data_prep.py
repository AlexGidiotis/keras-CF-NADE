import numpy as np

from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql import functions as F

from pyspark.mllib.linalg.distributed import CoordinateMatrix, MatrixEntry, BlockMatrix


from indexes import create_user_index, create_doc_index, load_indexes, map_recommendations

data_path = '/home/alex/Documents/Data/ml-1m/ratings.dat'








def read_data(sc,
	data_file,
	delimiter='::'):
	"""
	Read the data into an RDD of tuple (usrId, productId, rating).
	Args:
		sc: An active SparkContext.
		data_file: A (delimiter) separated file.
		delimiter: The delimiter used to separate the 3 fields of the input file. Default is ','.
	Returns:
		ui_mat_rdd: The UI matrix in an RDD.
	"""

	data = sc.textFile(data_file)
	header = data.first()
	ui_mat_rdd = data.filter(lambda row: row != header) \
		.map(lambda x: (int(x.split(delimiter)[0]),int(x.split(delimiter)[1]),float(x.split(delimiter)[2])))

	return ui_mat_rdd








if __name__ == "__main__":

	sc = SparkContext()
	spark = SparkSession(sc)

	ui_mat_rdd = read_data(sc, data_path, delimiter='::') \
		.sample(False,1.0,seed=0) \
		.persist()



	print 'Creating usr and doc indexes...'
	user_index = create_user_index(ui_mat_rdd)
	doc_index = create_doc_index(ui_mat_rdd)
	b_uidx = sc.broadcast(user_index)
	b_didx = sc.broadcast(doc_index)

	ui_mat_rdd = ui_mat_rdd.map(lambda (usrId,docId,value): (b_uidx.value[usrId],b_didx.value[docId],value))
		


	num_users = ui_mat_rdd.map(lambda (usrId,docId,value): usrId) \
		.distinct() \
		.count()
	num_movies = ui_mat_rdd.map(lambda (usrId,docId,value): docId) \
		.distinct() \
		.count()
	print 'users:',num_users,'products:',num_movies


	df = spark.createDataFrame(ui_mat_rdd,['userId','movieId','value'])

	ui_mat_rdd.unpersist()


	print 'Splitting data set...'
	df = df.orderBy(F.rand())

	train_df, test_df = df.randomSplit([0.9, 0.1], 
		seed=45)
	train_df, val_df = train_df.randomSplit([0.95, 0.05], 
		seed=45)

	train_size = train_df.count()
	val_size = val_df.count()
	test_size =  test_df.count()

	train_df.show()
	print train_size,'training examples'
	print val_size,'validation examples'
	print test_size,'testing example'

	
	train_input_ratings = np.zeros((num_movies, num_users), dtype='int8')
	train_output_ratings = np.zeros((num_movies, num_users), dtype='int8')
	#train_input_masks = np.zeros((num_movies, num_users), dtype='int8')
	#train_output_masks = np.zeros((n_movies, n_users), dtype='int8')
	
	valid_input_ratings = np.zeros((num_movies, num_users), dtype='int8')
	valid_output_ratings = np.zeros((num_movies, num_users), dtype='int8')
	#valid_input_masks = np.zeros((num_movies, num_users), dtype='int8')
	#valid_output_masks = np.zeros((n_movies, n_users), dtype='int8')
	
	test_input_ratings = np.zeros((num_movies, num_users), dtype='int8')
	test_output_ratings = np.zeros((num_movies, num_users), dtype='int8')
	#test_input_masks = np.zeros((num_movies, num_users), dtype='int8')
	#test_output_masks = np.zeros((n_movies, n_users), dtype='int8')

	for usr,mov,val in train_df.collect():
		train_input_ratings[mov,usr] = val
		valid_input_ratings[mov,usr] = val


	for usr,mov,val in val_df.collect():
		valid_input_ratings[mov,usr] = val

	for usr,mov,val in test_df.collect():
		test_input_ratings[mov,usr] = val

	print train_input_ratings
	print train_input_ratings.shape
	
	'''
	train_df, test_df = df.randomSplit([0.9, 0.1], seed=1)

	train_df \
		.repartition(200) \
		.persist()
	test_df \
		.repartition(200) \
		.persist()


	train_size = train_df.count()
	test_size = test_df.count()
	'''
