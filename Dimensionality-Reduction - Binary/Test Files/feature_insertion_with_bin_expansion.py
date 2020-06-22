from os.path import abspath, exists
import numpy as np
from Object_Files.mapper5 import mapper
from Object_Files.basic_operator import operator

import sys

import matplotlib.pyplot as plt
import random
import time

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
def get_adversarial_positions(demo_operator, batch_feature_size):
	feature_counter = demo_operator.get_feature_counter()
	
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
	return batch_positions

"""
    *
    * function load_data(file, number_of_objects)
    *
    * Summary: 
    *
    *   It reads the data from provided input file. We can give the
    *   limit to number of objects in the data with second parameter.
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
    *  If the feature exists, we take 1, otherwise 0. This way, it collects
    *   binary data out of a file.
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
            
            while True:
                
                line = f.readline() 
                if line:
                    words = line.split()
                    num = int(words[0])
                    position = int(words[1])
                    count = int(words[2])
                    
                    if num == last_num:
                        feature_array[position-1] = 1
                    else:
                        data_array.append(feature_array)
                        counter += 1
                        feature_array = np.zeros(features,dtype=int)
                        last_num = num
                        feature_array[position-1] = 1

                    if counter > number_of_objects :
                        break
		
                else:
                    break
    return data_array

"""
    *
    * function get_feature_insertion_results(Input_dimension, Output_dimension, default_bits, default_maps, array1, array2,mapping_scheme)
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
    *                  array1: Array of binary numbers 
    *                  array2: Array of binary numbers
    *                  mapping_scheme: Integer -- Note: Type of mapping used                                    
    *
    * Return Value  : batch_error: Array of real numbers (Error in original and predicted inner product)
    *                 batch_time: Array of real numbers (Time taken)
    *                 batch_inner_product1: Array of real numbers (Values of inner product of input arrays)
    *                 batch_inner_product2: Array of real numbers (Values of predicted inner product)
    *                 array1: Array of binary numbers (Compressed array of array1)
    *                 array2: Array of binary numbers (Compressed array of array2)
    *
    * Description:
    *
    *   This function inserts the numbers to given input array. Insertion with bin expansion is used.
    *   It then computes the affected output array and their inner products. 
    *   It finally returns all the results mentioned in return value section.
    *
"""
def get_feature_insertion_results(Input_dimension ,Output_dimension ,default_maps ,array1,array2,mapping_scheme=1,max_value=0):

    batch_error = []
    batch_time = []
    sample_size = Input_dimension/100
    increased_input_dim = int(Input_dimension*2)
    demo_operator = operator(input_dim=Input_dimension, output_dim=Output_dimension, mapping_scheme=mapping_scheme)
    ct = 0
    
    batch_inner_product1 = []
    batch_inner_product2 = []
    while Input_dimension <= increased_input_dim:
        print("\t", ct)
        ct+=1
        
        batch_feature_size = int(sample_size)
        batch_positions = get_adversarial_positions(demo_operator,batch_feature_size)
        feature1 = np.random.normal(0,1,size=batch_feature_size)
        feature2 = np.random.normal(0,1,size=batch_feature_size)

        Input_dimension+=batch_feature_size

        t1 = time.time()
        array1,array2 = demo_operator.batch_insert_feature(batch_positions,array1,array2, feature1, feature2)
        
        inner_product1, inner_product2 = demo_operator.inner_product(array1, array2)
        t2 = time.time()
        error = abs(inner_product1-inner_product2)
        
        batch_error.append(error)
        batch_inner_product1.append(inner_product1)
        batch_inner_product2.append(inner_product2)
        batch_time.append(t2-t1)
        
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
    *                  array1: Array of binary numbers 
    *                  array2: Array of binary numbers
    *                  mapping_scheme: Integer -- Note: Type of mapping used                                    
    *
    * Return Value  : batch_error: Array of real numbers (Error in original and predicted inner product)
    *                 batch_time: Array of real numbers (Time taken)
    *                 batch_inner_product1: Array of real numbers (Values of inner product of input arrays)
    *                 batch_inner_product2: Array of real numbers (Values of predicted inner product)
    *                 array1: Array of binary numbers (Compressed array of array1)
    *                 array2: Array of binary numbers (Compressed array of array2)
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
    
    batch_inner_product1 = []
    batch_inner_product2 = []
    while Input_dimension <= reduced_input_dim:
                
        batch_feature_size = int(sample_size)
        batch_positions = get_adversarial_positions(demo_operator,batch_feature_size)
        Input_dimension+=batch_feature_size
        feature1 = np.random.normal(0,1,size=batch_feature_size)
        feature2 = np.random.normal(0,1,size=batch_feature_size)

        t1 = time.time()
        array1,array2 = demo_operator.batch_insert_feature(batch_positions,array1,array2, feature1, feature2)
       
        fresh_operator = operator(input_dim=Input_dimension, output_dim=Output_dimension, mapping_scheme=mapping_scheme)
        inner_product1, inner_product2 = fresh_operator.inner_product(array1, array2)
        t2 = time.time()
        error = abs(inner_product1-inner_product2)
 
        batch_error.append(error)
        batch_time.append((t2-t1)*2)
        batch_inner_product1.append(inner_product1)
        batch_inner_product2.append(inner_product2)
                

    return batch_error, batch_time, batch_inner_product1,batch_inner_product2,array1,array2

"""
    *
    * function cumulate(arr)
    *
    * Summary: 
    *
    *   Given the array, it returns cumulative array of that.
    *   
    * Parameters     : arr: Array of real numbers 
    *                                                     
    *
    * Return Value  : temp_arr: Array of real numbers (cumulated)
    *
    *           -- Note: It also saves the numpy array of errors and time.
    *
    * Description:
    *
    *   Used in time taken array where we consider time before all deletions
    *
"""
def cumulate(arr):
    # temp_arr = np.zeros(len(arr))

    for i in range(1, len(arr)):
        arr[i] = arr[i-1]+arr[i]

    return arr

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

    *           -- Note: It also saves the numpy array of errors and time.
    *
    * Description:
    *
    *   This function is iterated over each dataset of collection of errors. Errors are also stored in file.
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
        
        maps = mapping.map

        print(count)
        
        alpha = 1

        arr1 = data_array[count-1]
        arr2 = data_array[count]

        norm_arr_1 = arr1 
        norm_arr_2 = arr2 
        
        batch_error_a, batch_time_a, batch_inner_product1_a,batch_inner_product2_a,_,_ = get_feature_insertion_results(Input_dimension = N,Output_dimension = M,default_maps=maps,array1=norm_arr_1,array2=norm_arr_2,mapping_scheme=5,max_value=alpha)

        batch_error_b, batch_time_b,batch_inner_product1_b,batch_inner_product2_b,_,_ = get_feature_insertion_results(Input_dimension = N,Output_dimension = M,default_maps=maps,array1=norm_arr_1,array2=norm_arr_2,mapping_scheme=6,max_value=alpha)

        batch_error_c, batch_time_c, batch_inner_product1_c,batch_inner_product2_c,_,_ = get_remap_results(Input_dimension = N,Output_dimension = M,array1=norm_arr_1,array2=norm_arr_2,mapping_scheme=5)
        
        if count == 1:
            avg_batch_error_a = batch_error_a
            avg_batch_error_b = batch_error_b
            avg_batch_error_c = batch_error_c

            avg_batch_time_a = batch_time_a
            avg_batch_time_b = batch_time_b
            avg_batch_time_c = batch_time_c


        else :
            for i in range(len(batch_error_a)):
                avg_batch_error_a[i] += batch_error_a[i]
                avg_batch_error_b[i] += batch_error_b[i]
                avg_batch_error_c[i] += batch_error_c[i]

                avg_batch_time_a[i] += batch_time_a[i]
                avg_batch_time_b[i] += batch_time_b[i]
                avg_batch_time_c[i] += batch_time_c[i]

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


    return avg_batch_error_a,  avg_batch_error_b, avg_batch_error_c, avg_batch_time_a, avg_batch_time_b, avg_batch_time_c

"""
    *
    * function main()
    *
    * Summary: 
    *
    *   Program initiation. 
    *   
    * Parameters     : None
    *                  command line argument(optional) : Number of data pairs                                 
    *
    * Return Value  : Nothing

    *           -- Note: Plots the errors and time and saves image in Plots folder.
    *
    * Description:
    *
    *   Iterates get_all_error function for all datasets. Plots results.
    *
"""	
def main():
    
    n_args = len(sys.argv)
    
    n_pairs = 100

    if n_args >1:
        n_pairs = int(sys.argv[1])

    files = {
        "NYtimes" :  "Data/docword.nytimes.txt", 
        "KOS" : "Data/docword.kos.txt", 
        "NIPS" : "Data/docword.nips.txt"
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
        print (x, "FINAL RESULTS",avg_batch_error_a,avg_batch_error_b,avg_batch_error_c)
        
        ax[0][it].plot(range(0,len(avg_batch_error_a)*10, 10), np.array(avg_batch_error_a)**2, label="No Compensation", linestyle='--')
        ax[0][it].plot(range(0,len(avg_batch_error_b)*10, 10), np.array(avg_batch_error_b)**2, label="Our Method", linewidth=3)
        ax[0][it].plot(range(0,len(avg_batch_error_c)*10, 10), np.array(avg_batch_error_c)**2, label="Remap")
        ax[0][it].legend(loc='upper right')

        ax[1][it].plot(range(0, len(avg_batch_time_a)*10, 10), cumulate(avg_batch_time_a), label="No Compensation", linestyle='--')
        ax[1][it].plot(range(0, len(avg_batch_time_b)*10, 10), cumulate(avg_batch_time_b), label="Our Method", linewidth=3)
        ax[1][it].plot(range(0, len(avg_batch_time_c)*10, 10), cumulate(avg_batch_time_c), label="Remap")
        ax[1][it].legend(loc='upper right')
        it+=1

    fig.tight_layout(pad=0.5)
    fig.set_figheight(8)
    fig.set_figwidth(12)


    fig.savefig('Plots/sample000_Insertion_1000.png', orientation = 'landscape')

if __name__ == '__main__':
	main()
