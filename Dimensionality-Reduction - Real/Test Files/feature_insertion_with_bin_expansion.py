from os.path import abspath, exists
import numpy as np
from Object_Files.mapper5 import mapper
from Object_Files.basic_operator import operator
#import matplotlib.pyplot as plt
import sys
# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import random
import time

"""
    *
    * function array_normalization(input_array)
    *
    * Summary: 
    *
    *   Given thr input array, this function returns a normalized array
    *   
    * Parameters     : input_array: Array of real numbers
    *
    * Return Value  : result: Array of real numbers
    *
    * Description:
    *
    *   We use this function to maintain the norm of array equal to 1.
    *
"""

def array_normalization(input_array):
    array_norm = np.linalg.norm(input_array)
    # print ("array norm:",array_norm)
    result = np.zeros(input_array.size, dtype=float)
    for i in range(input_array.size):
        result[i] = (1.0*input_array[i])/array_norm

    return result


"""
    *
    * function get_positions(demo_operator, batch_feature_size)
    *
    * Summary: 
    *
    *   For feature insertion, this function gives positions where
    *   features are going to be inserted.
    *   
    * Parameters     : demo_operator: Operator object
    *                  batch_feature_size: Integer                  
    *
    * Return Value  : batch_positions: Array of integers
    *
    * Description:
    *
    *   The function is useful while inserting the features at
    *   random positions.
    *
"""
def get_positions(demo_operator, batch_feature_size):
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

"""
    *
    * function load_data(file, number_of_objects)
    *
    * Summary: 
    *
    *   It reads the data from provied input file. We can give the
    *   limit to number of objects in the data ith second parameter.
    *   
    * Parameters     : file: String
    *                  number_of_objects: Integer                  
    *
    * Return Value  : data_array: Array of objects (data points)
    *
    * Description:
    *
    *   The file string contains absolute or relative path to the file.
    *   Number of objects are provided in order to limit the data size.
    *   When given file is not found, it returns nothing.
    *
"""
def load_data(file="Data/docword.enron.txt",number_of_objects = 100):
    data_array = []

    f_path = abspath(file)
    if exists(f_path):
        with open(f_path) as f:
            datapoints = int(f.readline())
            features = int(f.readline())
            unique_words = int(f.readline())
            last_num = 1
            feature_array = np.zeros(features,dtype=int)
            counter = 0
            # print("Count:",count)
            while True:
                
                line = f.readline() 
                if line:
                    words = line.split()
                    num = int(words[0])
                    position = int(words[1])
                    count = int(words[2])
                    # print(num)
                    if num == last_num:
                        feature_array[position-1] = count
                    else:
                        data_array.append(feature_array)
                        counter += 1
                        feature_array = np.zeros(features,dtype=int)
                        last_num = num
                        feature_array[position-1] = count

                    if counter > number_of_objects :
                        break



						
                else:
                    break
    return data_array

"""
    *
    * function get_feature_insertion_results(Input_dimension, Output_dimension, default_bits, default_maps, array1, array2,mapping_scheme, max_value)
    *
    * Summary: 
    *
    *   This function inserts the features and returns error and other values.
    *   Input arrays are taken as a parameter.
    *   
    * Parameters     : Input_dimension: Integer
    *                  Output_dimension: Integer
    *                  default_bits: Array of bool
    *                  default_maps: Array of integers
    *                  array1: Array of real numbers 
    *                  array2: Array of real numbers
    *                  mapping_scheme: Integer -- Note: Type of mapping used                                    
    *
    * Return Value  : batch_error: Array of real numbers (Error in original and predicted inner product)
    *                 batch_time: Array of real numbers (Time taken)
    *                 batch_inner_product1: Array of real numbers (Values of inner product of input arrays)
    *                 batch_inner_product2: Array of real numbers (Values of predicted inner product)
    *                 array1: Array of real numbers (Compressed array of array1)
    *                 array2: Array of real numbers (Compressed array of array2)
    *
    * Description:
    *
    *   This function inserts the numbers to given input array. Insertion with bin expansion is used.
    *   It then computes the affected output array and their inner products. 
    *   It finally returns all the results mentioned in return value section.
    *
"""
def get_feature_insertion_results(Input_dimension ,Output_dimension ,default_bits ,default_maps ,array1,array2,mapping_scheme=1,max_value=0):

    batch_error = []
    batch_time = []
    sample_size = Input_dimension/100
    increased_input_dim = int(Input_dimension*2)
    demo_operator = operator(input_dim=Input_dimension, output_dim=Output_dimension, mapping_scheme=mapping_scheme)
    ct = 0
    # demo_operator.mapping.bits = default_bits
    # demo_operator.mapping.map = default_maps
    batch_inner_product1 = []
    batch_inner_product2 = []
    while Input_dimension <= increased_input_dim:
        print("\t", ct)
        ct+=1
        # print ("epoch1:::Input Dimenson::",Input_dimension)
        batch_feature_size = int(sample_size)
        batch_positions = get_positions(demo_operator,batch_feature_size)
        feature1 = np.random.normal(0,1,size=batch_feature_size)
        feature2 = np.random.normal(0,1,size=batch_feature_size)



        Input_dimension+=batch_feature_size

        t1 = time.time()
        array1,array2 = demo_operator.batch_insert_feature(batch_positions,array1,array2, feature1, feature2)
        # print("batch feature insertion done....")
        # print("arr1:",array1)
        # print("arr2:",array2)
        inner_product1, inner_product2 = demo_operator.inner_product(array1, array2)
        t2 = time.time()
        error = abs(inner_product1-inner_product2)
        # print ("inners products:",inner_product1,inner_product2)
        # print("error:", error)
        batch_error.append(error)
        batch_inner_product1.append(inner_product1)
        batch_inner_product2.append(inner_product2)
        batch_time.append(t2-t1)
        # print ("Mapping scheme :",mapping_scheme,"::")
        # print (demo_operator.get_feature_count())
		

    return batch_error, batch_time, batch_inner_product1,batch_inner_product2,array1,array2

"""
    *
    * function get_remap_results(Input_dimension, Output_dimension, array1, array2, mapping_scheme)
    *
    * Summary: 
    *
    *   This function inserts the features and returns error and other values.
    *   Input arrays are taken as a parameter.
    *   
    * Parameters     : Input_dimension: Integer
    *                  Output_dimension: Integer
    *                  array1: Array of real numbers 
    *                  array2: Array of real numbers
    *                  mapping_scheme: Integer -- Note: Type of mapping used                                    
    *
    * Return Value  : batch_error: Array of real numbers (Error in original and predicted inner product)
    *                 batch_time: Array of real numbers (Time taken)
    *                 batch_inner_product1: Array of real numbers (Values of inner product of input arrays)
    *                 batch_inner_product2: Array of real numbers (Values of predicted inner product)
    *                 array1: Array of real numbers (Compressed array of array1)
    *                 array2: Array of real numbers (Compressed array of array2)
    *
    * Description:
    *
    *   This function works similar to the above function. After insertion of features in
    *   input arrays, it creates fresh mapping for modified array (by creating new operator object).
    *
"""
def get_remap_results(Input_dimension, Output_dimension, array1, array2, mapping_scheme):
    batch_error = []
    batch_time = []
    sample_size = Input_dimension/100
    reduced_input_dim = int(Input_dimension*2)
    demo_operator = operator(input_dim=Input_dimension, output_dim=Output_dimension, mapping_scheme=mapping_scheme)
    # demo_operator.mapping.bits = default_bits
    # demo_operator.mapping.map = default_maps
    batch_inner_product1 = []
    batch_inner_product2 = []
    while Input_dimension <= reduced_input_dim:
        # print ("epoch1:::Input Dimension::",Input_dimension)
        
        batch_feature_size = int(sample_size)
        batch_positions = get_positions(demo_operator,batch_feature_size)
        Input_dimension+=batch_feature_size
        feature1 = np.random.normal(0,1,size=batch_feature_size)
        feature2 = np.random.normal(0,1,size=batch_feature_size)

        t1 = time.time()
        array1,array2 = demo_operator.batch_insert_feature(batch_positions,array1,array2, feature1, feature2)
        # print("batch feature insertion done....")
        # print("arr1:",array1)
        # print("arr2:",array2)
        fresh_operator = operator(input_dim=Input_dimension, output_dim=Output_dimension, mapping_scheme=mapping_scheme)
        inner_product1, inner_product2 = fresh_operator.inner_product(array1, array2)
        t2 = time.time()
        error = abs(inner_product1-inner_product2)
        # print ("inners products:",inner_product1,inner_product2)
        # print("error:", error)
        batch_error.append(error)
        batch_time.append((t2-t1)*2)
        batch_inner_product1.append(inner_product1)
        batch_inner_product2.append(inner_product2)
        # print ("Mapping scheme :",mapping_scheme,"::")
        # print (demo_operator.get_feature_count())
        

    return batch_error, batch_time, batch_inner_product1,batch_inner_product2,array1,array2

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

"""
    *
    * function get_all_errors(input_file, n_pairs)
    *
    * Summary: 
    *
    *   This function collects all the errors and time taken for given dataset 
    *   file name (path), and returns average of them.
    *   
    * Parameters     : input_file: String
    *                  n_pairs: Integer                                    
    *
    * Return Value  : avg_batch_error_a: Array of real numbers (Average error of no compensation)
    *                 avg_batch_error_b: Array of real numbers (Average error of our method)
    *                 avg_batch_error_c: Array of real numbers (Average error of total remap)
    *                 avg_batch_time_a: Array of real numbers (Average time taken for no compensation)
    *                 avg_batch_time_b: Array of real numbers (Average time taken for our method)
    *                 avg_batch_time_c: Array of real numbers (Average time taken for total remap)
    *
    * Description:
    *
    *   This function works similar to the above function. After insertion of features in
    *   input arrays, it creates fresh mapping for modified array (by creating new operator object).
    *
"""		
def get_all_errors(input_file, n_pairs, compensation1, compensation2):
    count = 1
    avg_batch_error_a = []
    avg_batch_error_b = []
    avg_batch_error_c = []
    
    avg_batch_time_a = []
    avg_batch_time_b = []
    avg_batch_time_c = []
    data_array = load_data(input_file,n_pairs)
    print(data_array)
    N = data_array[0].size
    M = 2000

    dataset = input_file.split('.')[1]
    
    while count < n_pairs-1:
        
        mapping = mapper(N,M)
        bits = mapping.bits
        maps = mapping.map

        print(count)
        
        # print ("* Input Dimension of Dataset:",N)
        # print ("* Output (compressed) Dimension of Dataset:",M)
        alpha = 1

        arr1 = data_array[count-1]
        arr2 = data_array[count]

        # print ("* Selected array (1) from Dataset:",arr1)
        # print ("* Selected array (2) from Dataset:",arr2)

        norm_arr_1 = array_normalization(arr1)
        norm_arr_2 = array_normalization(arr2)

        # norm_arr_1 = arr1
        # norm_arr_2 = arr2

        # print ("* Normalized array (1):",norm_arr_1)
        # print ("* Normalized array (2):",norm_arr_2)

        batch_error_a, batch_time_a, batch_inner_product1_a,batch_inner_product2_a,_,_ = get_feature_insertion_results(Input_dimension = N,Output_dimension = M,default_bits=bits,default_maps=maps,array1=norm_arr_1,array2=norm_arr_2,mapping_scheme=5,max_value=alpha)

        # plt.plot(range(len(batch_error)), batch_error, label = "Error Without Compensation")
        # plt.plot(range(len(batch_inner_product1)), batch_inner_product1, label = "IP1 Without Compensation")
        # plt.plot(range(len(batch_inner_product2)), batch_inner_product2, label = "IP2 Without Compensation")

        batch_error_b, batch_time_b,batch_inner_product1_b,batch_inner_product2_b,_,_ = get_feature_insertion_results(Input_dimension = N,Output_dimension = M,default_bits=bits,default_maps=maps,array1=norm_arr_1,array2=norm_arr_2,mapping_scheme=6,max_value=alpha)

        batch_error_c, batch_time_c, batch_inner_product1_c,batch_inner_product2_c,_,_ = get_remap_results(Input_dimension = N,Output_dimension = M,array1=norm_arr_1,array2=norm_arr_2,mapping_scheme=6)
        # batch_error_c,batch_inner_product1_c,batch_inner_product2_c,_,_ = get_feature_insertion_results(Input_dimension = N,Output_dimension = M,default_bits=bits,default_maps=maps,array1=norm_arr_1,array2=norm_arr_2,mapping_scheme=8,max_value=alpha)

        # print(batch_error,batch_inner_product1,batch_inner_product2,array1,array2)

        # plt.plot(range(len(batch_error)), batch_error, label = "Error With Compensation")
        # plt.plot(range(len(batch_inner_product1)), batch_inner_product1, label = "IP1 With Compensation")
        # plt.plot(range(len(batch_inner_product2)), batch_inner_product2, label = "IP2 With Compensation")
        # plt.legend()
        # plt.show()
        if count == 1:
            avg_batch_error_a = batch_error_a
            avg_batch_error_b = batch_error_b
            avg_batch_error_c = batch_error_c

            avg_batch_time_a = batch_time_a
            avg_batch_time_b = batch_time_b
            avg_batch_time_c = batch_time_c

            # avg_inner_product1_a = batch_inner_product1_a
            # avg_inner_product2_a = batch_inner_product2_a
            # avg_inner_product1_b = batch_inner_product1_b
            # avg_inner_product2_b = batch_inner_product2_b
            # avg_inner_product1_c = batch_inner_product1_c
            # avg_inner_product2_c = batch_inner_product2_c

        else :
            for i in range(len(batch_error_a)):
                avg_batch_error_a[i] += batch_error_a[i]
                avg_batch_error_b[i] += batch_error_b[i]
                avg_batch_error_c[i] += batch_error_c[i]

                avg_batch_time_a[i] += batch_time_a[i]
                avg_batch_time_b[i] += batch_time_b[i]
                avg_batch_time_c[i] += batch_time_c[i]

                # avg_inner_product1_a[i] += batch_inner_product1_a[i]
                # avg_inner_product2_a[i] += batch_inner_product2_a[i]
                # avg_inner_product1_b[i] += batch_inner_product1_b[i]
                # avg_inner_product2_b[i] += batch_inner_product2_b[i]
                # avg_inner_product1_c[i] += batch_inner_product1_c[i]
                # avg_inner_product2_c[i] += batch_inner_product2_c[i]

        if count%50 == 0 or count == n_pairs-2:
            np.save('Outputs/insertion/sample0000_testing_'+dataset+'_'+str(count)+'.npy', [ (np.array(avg_batch_error_a)/count, np.array(batch_time_a)/count), (np.array(avg_batch_error_b)/count, np.array(batch_time_b)/count), (np.array(avg_batch_error_c)/count, np.array(batch_time_c)/count) ])

        count += 1

    for i in range(len(avg_batch_error_a)):
        avg_batch_error_a[i] /= n_pairs
        avg_batch_error_b[i] /= n_pairs
        avg_batch_error_c[i] /= n_pairs

        avg_batch_time_a[i] /= n_pairs
        avg_batch_time_b[i] /= n_pairs
        avg_batch_time_c[i] /= n_pairs

        # avg_inner_product1_a[i] /= n_pairs
        # avg_inner_product2_a[i] /= n_pairs
        # avg_inner_product1_b[i] /= n_pairs
        # avg_inner_product2_b[i] /= n_pairs
        # avg_inner_product1_c[i] /= n_pairs
        # avg_inner_product2_c[i] /= n_pairs

    return avg_batch_error_a,  avg_batch_error_b, avg_batch_error_c, avg_batch_time_a, avg_batch_time_b, avg_batch_time_c

def main():
    # input_file = sys.argv[1]
    # compensation1, compensation2, compensation3 = 0, 1, 2 # 0 = No Compensaation, 1 = 1 step Compensation, 2 = 2 step
    n_args = len(sys.argv)
    # if n_args > 2:
    #     compensation1 = int(sys.argv[2])
    #     compensation2 = int(sys.argv[3])
    
    # m1, m2 = 5, 6 #One without compensation, other with one step compensation
    # if compensation1 == 0:
    #     m1 = 5
    # elif compensation1 == 1:
    #     m1 = 6
    # else:
    #     m1 = 8

    # if compensation2 == 0:
    #     m2 = 5
    # elif compensation2 == 1:
    #     m2 = 6
    # else:
    #     m2 = 8

    n_pairs = 100

    if n_args >1:
        n_pairs = int(sys.argv[1])

    files = {
        "NYtimes" :  "Data/docword.nytimes.txt", #"./Data/docword.enron.txt", #
        "KOS" : "Data/docword.kos.txt", #"./Data/docword.kos.txt", #
        "NIPS" : "Data/docword.nips.txt" #"./Data/docword.nips.txt", #
    }

    fig, ax = plt.subplots(2, 3)
    
    ax[0][0].set_title('NYtimes')
    ax[0][1].set_title('KOS')
    ax[0][2].set_title('NIPS')

    ax[1][0].set_title('NYtimes')
    ax[1][1].set_title('KOS')
    ax[1][2].set_title('NIPS')
    

    ax[0][0].set(xlabel='% of features inserted', ylabel='MSE')
    ax[0][1].set(xlabel='% of features inserted', ylabel='MSE')
    ax[0][2].set(xlabel='% of features inserted', ylabel='MSE')

    ax[1][0].set(xlabel='% of features inserted', ylabel='Time(s)')
    ax[1][1].set(xlabel='% of features inserted', ylabel='Time(s)')
    ax[1][2].set(xlabel='% of features inserted', ylabel='Time(s)')

    #loop
    it = 0
    for x, y in files.items():
        print("Dataset:", x)
        avg_batch_error_a, avg_batch_error_b, avg_batch_error_c, avg_batch_time_a, avg_batch_time_b, avg_batch_time_c = get_all_errors(y, n_pairs, 5, 6)
        print ("FINAL RESULTS",avg_batch_error_a,avg_batch_error_b,avg_batch_error_c)
        # avg_batch_error_a, avg_batch_error_b, avg_batch_error_c, avg_batch_time_a, avg_batch_time_b, avg_batch_time_c = [1,2,4,5,6,8,9,1,4,6], [2,2,4,5,6,8,9,1,4,6], [3,2,4,5,6,8,9,1,4,6], [4,2,4,5,6,8,9,1,4,6], [5,2,4,5,6,8,9,1,4,6], [6,2,4,5,6,8,9,1,4,6]
    
    

    
        print(avg_batch_error_a)
        print(avg_batch_error_b)
        print(avg_batch_error_c)
        ax[0][it].plot(range(0,len(avg_batch_error_a)*10, 10), np.array(avg_batch_error_a)**2, label="No Compensation", linestyle='--')
        ax[0][it].plot(range(0,len(avg_batch_error_b)*10, 10), np.array(avg_batch_error_b)**2, label="Our Method", linewidth=3)
        ax[0][it].plot(range(0,len(avg_batch_error_c)*10, 10), np.array(avg_batch_error_c)**2, label="Remap")
        ax[0][it].legend(loc='upper right')

        print(avg_batch_time_a)
        print(avg_batch_time_b)
        print(avg_batch_time_c)
        ax[1][it].plot(range(0, len(avg_batch_time_a)*10, 10), avg_batch_time_a, label="No Compensation", linestyle='--')
        ax[1][it].plot(range(0, len(avg_batch_time_b)*10, 10), avg_batch_time_b, label="Our Method", linewidth=3)
        ax[1][it].plot(range(0, len(avg_batch_time_c)*10, 10), avg_batch_time_c, label="Remap")
        ax[1][it].legend(loc='upper right')
        it+=1

    


    # fig.legend()
    fig.tight_layout(pad=0.5)
    fig.set_figheight(8)
    fig.set_figwidth(12)

    
    #plt.show()
    # fig.savefig("./fig.png")
    fig.savefig('Plots/sample000_Insertion_1000.png', orientation = 'landscape')

    # return

    # plt.plot(range(len(avg_batch_error_a)), np.array(avg_batch_error_a)**2, label = "NO Compensation")
    # # plt.plot(range(len(avg_inner_product1_a)), avg_inner_product1_a, label = "IP1 With "+str(compensation1)+" step Compensation")
    # # plt.plot(range(len(avg_inner_product2_a)), avg_inner_product2_a, label = "IP2 With "+str(compensation1)+" step Compensation")

    # plt.plot(range(len(avg_batch_error_b)), np.array(avg_batch_error_b)**2, label = "One Step Compensation")
    # # plt.plot(range(len(avg_inner_product1_b)), avg_inner_product1_b, label = "IP1 With "+str(compensation2)+" step Compensation")
    # # plt.plot(range(len(avg_inner_product2_b)), avg_inner_product2_b, label = "IP2 With "+str(compensation2)+" step Compensation")

    # plt.plot(range(len(avg_batch_error_c)), np.array(avg_batch_error_c)**2, label = "Remap")
    # # plt.plot(range(len(avg_inner_product1_c)), avg_inner_product1_c, label = "IP1 With "+str(compensation3)+" step Compensation")
    # # plt.plot(range(len(avg_inner_product2_c)), avg_inner_product2_c, label = "IP2 With "+str(compensation3)+" step Compensation")
    # plt.xlabel("% of features deleted")
    # plt.ylabel("MSE")
    # plt.legend()

    # #plt.show()
    # plt.savefig('/home/b16032/MTP/Dimensionality-Reduction/Test Files/Plots/All_Datasets_28-2-2020.png')



    # batch_error,batch_inner_product1,batch_inner_product2,_,_ = get_feature_insertion_results(Input_dimension = N,Output_dimension = M,array1=norm_arr_1,array2=norm_arr_2,mapping_scheme=3,max_value=alpha)

    # plt.plot(range(len(batch_error)), batch_error, label = "Error Without Compensation")
    # plt.plot(range(len(batch_inner_product1)), batch_inner_product1, label = "IP1 Without Compensation")
    # plt.plot(range(len(batch_inner_product2)), batch_inner_product2, label = "IP2 Without Compensation")

    # batch_error,batch_inner_product1,batch_inner_product2,_,_ = get_feature_insertion_results(Input_dimension = N,Output_dimension = M,array1=norm_arr_1,array2=norm_arr_2,mapping_scheme=4,max_value=alpha)

    # # print(batch_error,batch_inner_product1,batch_inner_product2,array1,array2)

    # plt.plot(range(len(batch_error)), batch_error, label = "Error With Compensation")
    # plt.plot(range(len(batch_inner_product1)), batch_inner_product1, label = "IP1 With Compensation")
    # plt.plot(range(len(batch_inner_product2)), batch_inner_product2, label = "IP2 With Compensation")
    # plt.legend()
    # plt.show()

if __name__ == '__main__':
	main()
