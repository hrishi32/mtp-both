from os.path import abspath, exists
import numpy as np
from Object_Files.mapper5 import mapper
from Object_Files.basic_operator import operator
import matplotlib.pyplot as plt
import random

mapping = mapper(28102,2000)
default_bits = mapping.bits
default_maps = mapping.map

def array_normalization(input_array):
    array_norm = np.linalg.norm(input_array)
    # print ("array norm:",array_norm)
    result = np.zeros(input_array.size, dtype=float)
    for i in range(input_array.size):
        result[i] = (1.0*input_array[i])/array_norm

    return result


def get_adversarial_positions(demo_operator, batch_feature_size):
	feature_counter = demo_operator.get_feature_counter()
	# print ("Originl feature counter:",feature_counter)
	batch_positions = []
	alpha_map = np.zeros(len(feature_counter))
	while len(batch_positions) < batch_feature_size:
		alpha = random.randint(0, len(feature_counter)-1)
		if alpha_map[alpha] == 1:
			continue
		else :
			alpha_map[alpha] = 1

		for i in feature_counter[alpha]:
			if len(batch_positions) < batch_feature_size:
				batch_positions.append(i)
			else:
				break
		
	batch_positions.sort()
	# print ("batch positions to be deleted:",batch_positions)
	return batch_positions

def load_data():
	data_array = []

	f_path = abspath("docword.enron.txt")
	if exists(f_path):
		with open(f_path) as f:
			datapoints = int(f.readline())
			features = int(f.readline())
			unique_words = int(f.readline())
			last_num = 1
			feature_array = np.zeros(features,dtype=int)
			count = 0
			while True:
				line = f.readline() 
				if line:
					words = line.split()
					num = int(words[0])
					position = int(words[1])
					count = int(words[2])
					if num == last_num:
						feature_array[position-1] = count
					else:
						data_array.append(feature_array)
						count += 1
						feature_array = np.zeros(features,dtype=int)
						last_num = num
						feature_array[position-1] = count

					if count > 100 :
						break



						
				else:
					break
	return data_array

def get_feature_deletion_results(Input_dimension ,Output_dimension ,array1,array2,mapping_scheme=1,max_value=0):

	batch_error = []
	sample_size = Input_dimension/100
	reduced_input_dim = Input_dimension/1.5
	demo_operator = operator(input_dim=Input_dimension, output_dim=Output_dimension, mapping_scheme=mapping_scheme)
	demo_operator.mapping.bits = default_bits
	demo_operator.mapping.map = default_maps
	batch_inner_product1 = []
	batch_inner_product2 = []
	while Input_dimension >= reduced_input_dim:
		# print ("epoch1:::Input Dimenson::",Input_dimension)
		batch_feature_size = int(sample_size)
		batch_positions = get_adversarial_positions(demo_operator,batch_feature_size)
		Input_dimension-=batch_feature_size

		array1,array2 = demo_operator.batch_delete_feature(batch_positions,array1,array2)
        # print("batch feature deletion done....")
        # print("arr1:",array1)
        # print("arr2:",array2)
		inner_product1, inner_product2 = demo_operator.inner_product(array1, array2)
		error = abs(inner_product1-inner_product2)
		print ("inners products:",inner_product1,inner_product2)
		print("error:", error)
		batch_error.append(error)
		batch_inner_product1.append(inner_product1)
		batch_inner_product2.append(inner_product2)
		# print ("Mapping scheme :",mapping_scheme,"::")
		# print (demo_operator.get_feature_count())
		

	return batch_error,batch_inner_product1,batch_inner_product2,array1,array2

def get_inner_product_results(array1, array2, input_dimension, output_dimension):
	i = 10
	avg_inner_product1, avg_inner_product2 = 0, 0
	while i > 0:
		demo_operator = operator(input_dim=input_dimension, output_dim=output_dimension, mapping_scheme = 3)
		inner_product1, inner_product2 = demo_operator.inner_product(array1, array2)

		avg_inner_product1+=inner_product1
		avg_inner_product2+=inner_product2

		i-=1
	
	avg_inner_product1/=10
	avg_inner_product2/=10

	return avg_inner_product1, avg_inner_product2
		


def main():
	count = 1
	avg_batch_error_a = []
	avg_batch_error_b = []
	avg_inner_product1_a = []
	avg_inner_product2_a = []
	avg_inner_product1_b = []
	avg_inner_product2_b = []
	while count <= 100:

		data_array = load_data()
		N = data_array[0].size
		M = 2000
		# print ("* Input Dimension of Dataset:",N)
		# print ("* Output (compressed) Dimension of Dataset:",M)
		alpha = 1

		arr1 = data_array[count-1]
		arr2 = data_array[count]

		# print ("* Selected array (1) from Dataset:",arr1)
		# print ("* Selected array (2) from Dataset:",arr2)

		# norm_arr_1 = array_normalization(arr1)
		# norm_arr_2 = array_normalization(arr2)

		norm_arr_1 = arr1
		norm_arr_2 = arr2

		# print ("* Normalized array (1):",norm_arr_1)
		# print ("* Normalized array (2):",norm_arr_2)

		batch_error_a,batch_inner_product1_a,batch_inner_product2_a,_,_ = get_feature_deletion_results(Input_dimension = N,Output_dimension = M,array1=norm_arr_1,array2=norm_arr_2,mapping_scheme=5,max_value=alpha)

		# plt.plot(range(len(batch_error)), batch_error, label = "Error Without Compensation")
		# plt.plot(range(len(batch_inner_product1)), batch_inner_product1, label = "IP1 Without Compensation")
		# plt.plot(range(len(batch_inner_product2)), batch_inner_product2, label = "IP2 Without Compensation")

		batch_error_b,batch_inner_product1_b,batch_inner_product2_b,_,_ = get_feature_deletion_results(Input_dimension = N,Output_dimension = M,array1=norm_arr_1,array2=norm_arr_2,mapping_scheme=6,max_value=alpha)

		# print(batch_error,batch_inner_product1,batch_inner_product2,array1,array2)

		# plt.plot(range(len(batch_error)), batch_error, label = "Error With Compensation")
		# plt.plot(range(len(batch_inner_product1)), batch_inner_product1, label = "IP1 With Compensation")
		# plt.plot(range(len(batch_inner_product2)), batch_inner_product2, label = "IP2 With Compensation")
		# plt.legend()
		# plt.show()
		if count == 1:
			avg_batch_error_a = batch_error_a
			avg_batch_error_b = batch_error_b
			avg_inner_product1_a = batch_inner_product1_a
			avg_inner_product2_a = batch_inner_product2_a
			avg_inner_product1_b = batch_inner_product1_b
			avg_inner_product2_b = batch_inner_product2_b

<<<<<<< HEAD
		else :
			for i in range(len(batch_error_a)):
				avg_batch_error_a[i] += batch_error_a[i]
				avg_batch_error_b[i] += batch_error_b[i]
				avg_inner_product1_a[i] += batch_inner_product1_a[i]
				avg_inner_product2_a[i] += batch_inner_product2_a[i]
				avg_inner_product1_b[i] = batch_inner_product1_b[i]
				avg_inner_product2_b[i] = batch_inner_product2_b[i]
		count += 1

	for i in range(len(avg_batch_error_a)):
		avg_batch_error_a[i] /= 100
		avg_batch_error_b[i] /= 100
		avg_inner_product1_a[i] /= 100
		avg_inner_product2_a[i] /= 100
		avg_inner_product1_b[i] /= 100
		avg_inner_product2_b[i] /= 100

	plt.plot(range(len(avg_batch_error_a)), avg_batch_error_a, label = "Error Without Compensation")
	plt.plot(range(len(avg_inner_product1_a)), avg_inner_product1_a, label = "IP1 Without Compensation")
	plt.plot(range(len(avg_inner_product2_a)), avg_inner_product2_a, label = "IP2 Without Compensation")

	plt.plot(range(len(avg_batch_error_b)), avg_batch_error_b, label = "Error With Compensation")
	plt.plot(range(len(avg_inner_product1_b)), avg_inner_product1_b, label = "IP1 With Compensation")
	plt.plot(range(len(avg_inner_product2_b)), avg_inner_product2_b, label = "IP2 With Compensation")
=======
	data_array = load_data()
	i=10
	inner_products1, inner_products2 = [], []
	errors = []
	while i > 0:
		N = data_array[0].size
		M = 50*(11-i)
		print ("* Input Dimension of Dataset:",N)
		print ("* Output (compressed) Dimension of Dataset:",M)
		alpha = 1

		arr1 = data_array[0]
		arr2 = data_array[1]

		print ("* Selected array (1) from Dataset:",arr1)
		print ("* Selected array (2) from Dataset:",arr2)

		# norm_arr_1 = array_normalization(arr1)
		# norm_arr_2 = array_normalization(arr2)

		norm_arr_1 = arr1
		norm_arr_2 = arr2

		print ("* Normalized array (1):",norm_arr_1)
		print ("* Normalized array (2):",norm_arr_2)

		ip1, ip2 = get_inner_product_results(norm_arr_1, norm_arr_2, N, M)

		inner_products1.append(ip1)
		inner_products2.append(ip2)

		errors.append(abs(ip1-ip2))

		i-=1

	plt.plot(range(10), inner_products1, label="Inner Product 1")
	plt.plot(range(10), inner_products2, label = "Inner Product 2")
	plt.plot(range(10), errors, label = "Errors")
>>>>>>> c1baf7dafdaf57e4983d7eb7c9d456fad83b174b
	plt.legend()

	plt.show()
	


	# batch_error,batch_inner_product1,batch_inner_product2,_,_ = get_feature_deletion_results(Input_dimension = N,Output_dimension = M,array1=norm_arr_1,array2=norm_arr_2,mapping_scheme=3,max_value=alpha)

	# plt.plot(range(len(batch_error)), batch_error, label = "Error Without Compensation")
	# plt.plot(range(len(batch_inner_product1)), batch_inner_product1, label = "IP1 Without Compensation")
	# plt.plot(range(len(batch_inner_product2)), batch_inner_product2, label = "IP2 Without Compensation")

	# batch_error,batch_inner_product1,batch_inner_product2,_,_ = get_feature_deletion_results(Input_dimension = N,Output_dimension = M,array1=norm_arr_1,array2=norm_arr_2,mapping_scheme=4,max_value=alpha)

	# # print(batch_error,batch_inner_product1,batch_inner_product2,array1,array2)

	# plt.plot(range(len(batch_error)), batch_error, label = "Error With Compensation")
	# plt.plot(range(len(batch_inner_product1)), batch_inner_product1, label = "IP1 With Compensation")
	# plt.plot(range(len(batch_inner_product2)), batch_inner_product2, label = "IP2 With Compensation")
	# plt.legend()
	# plt.show()

if __name__ == '__main__':
	main()
