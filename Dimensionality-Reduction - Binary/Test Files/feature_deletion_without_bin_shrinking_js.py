from os.path import abspath, exists
import numpy as np
from Object_Files.mapper5 import mapper
from Object_Files.basic_operator import operator

import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import random
import time

"""
    *
    * function get_adversarial_positions(demo_operator, batch_feature_size)
    *
    * Summary: 
    *
    *   For feature deletion, this function gives positions where
    *   features are going to be deleted.
    *   
    * Parameters     : demo_operator: Operator object
    *                  batch_feature_size: Integer                  
    *
    * Return Value  : batch_positions: Array of integers
    *
    * Description:
    *
    *   As the deletion method is 'adverserial deletion', this function picks a bin 
    *   and remove all the elements until batch_feature_size number of elements are deleted.
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
	# print ("batch positions to be deleted:",batch_positions)
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
    *   If the feature exists, we take 1, otherwise 0. This way, it collects
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
    * function get_feature_deletion_results(Input_dimension, Output_dimension, default_bits, default_maps, array1, array2,mapping_scheme)
    *
    * Summary: 
    *
    *   This function deletes the features and returns error and other values.
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
    * Return Value  : batch_error: Array of real numbers (Error in original and predicted jaccard similarity)
    *                 batch_time: Array of real numbers (Time taken)
    *                 batch_inner_product1: Array of real numbers (Values of jaccard similarity of input arrays)
    *                 batch_inner_product2: Array of real numbers (Values of predicted jaccard similarity)
    *                 array1: Array of binary numbers (Compressed array of array1)
    *                 array2: Array of binary numbers (Compressed array of array2)
    *
    * Description:
    *
    *   This function delets the numbers to given input array. Deletion without bin shrinking is used.
    *   It then computes the affected output array and their jaccard similaritys. 
    *   It finally returns all the results mentioned in return value section.
    *
"""
def get_feature_deletion_results(Input_dimension ,Output_dimension ,default_maps ,array1,array2,mapping_scheme=1,max_value=0):

    batch_error = []
    batch_time = []
    sample_size = Input_dimension/100
    reduced_input_dim = Input_dimension//2
    demo_operator = operator(input_dim=Input_dimension, output_dim=Output_dimension, mapping_scheme=mapping_scheme)
    
    # demo_operator.mapping.map = default_maps
    batch_jaccard_similarity1 = []
    batch_jaccard_similarity2 = []
    while Input_dimension >= reduced_input_dim:
        
        batch_feature_size = int(sample_size)
        batch_positions = get_adversarial_positions(demo_operator,batch_feature_size)
        Input_dimension-=batch_feature_size

        t1 = time.time()
        array1,array2 = demo_operator.batch_delete_feature(batch_positions,array1,array2)
        
        jaccard_similarity1, jaccard_similarity2 = demo_operator.jaccard_similarity(array1, array2)
        t2 = time.time()
        error = abs(jaccard_similarity1-jaccard_similarity2)
        
        batch_error.append(error)
        batch_jaccard_similarity1.append(jaccard_similarity1)
        batch_jaccard_similarity2.append(jaccard_similarity2)
        batch_time.append(t2-t1)
        
    return batch_error, batch_time, batch_jaccard_similarity1,batch_jaccard_similarity2,array1,array2

"""
    *
    * function get_remap_results(Input_dimension, Output_dimension, array1, array2, mapping_scheme)
    *
    * Summary: 
    *
    *   This function delets the features and returns error and other values.
    *   Input arrays are taken as a parameter.
    *   
    * Parameters     : Input_dimension: Integer
    *                  Output_dimension: Integer
    *                  array1: Array of binary numbers 
    *                  array2: Array of binary numbers
    *                  mapping_scheme: Integer -- Note: Type of mapping used                                    
    *
    * Return Value  : batch_error: Array of real numbers (Error in original and predicted jaccard similarity)
    *                 batch_time: Array of real numbers (Time taken)
    *                 batch_inner_product1: Array of real numbers (Values of jaccard similarity of input arrays)
    *                 batch_inner_product2: Array of real numbers (Values of predicted jaccard similarity)
    *                 array1: Array of binary numbers (Compressed array of array1)
    *                 array2: Array of binary numbers (Compressed array of array2)
    *
    * Description:
    *
    *   This function works similar to the above function. After deletion of features in
    *   input arrays, it creates fresh mapping for modified array (by creating new operator object).
    *
"""
def get_remap_results(Input_dimension, Output_dimension, array1, array2, mapping_scheme):
    batch_error = []
    batch_time = []
    sample_size = Input_dimension/100
    reduced_input_dim = Input_dimension//2
    demo_operator = operator(input_dim=Input_dimension, output_dim=Output_dimension, mapping_scheme=mapping_scheme)
    
    batch_jaccard_similarity1 = []
    batch_jaccard_similarity2 = []
    while Input_dimension >= reduced_input_dim:
        
        batch_feature_size = int(sample_size)
        batch_positions = get_adversarial_positions(demo_operator,batch_feature_size)
        Input_dimension-=batch_feature_size

        t1 = time.time()
        array1,array2 = demo_operator.batch_delete_feature(batch_positions,array1,array2)
        
        fresh_operator = operator(input_dim=Input_dimension, output_dim=Output_dimension, mapping_scheme=mapping_scheme)
        jaccard_similarity1, jaccard_similarity2 = fresh_operator.jaccard_similarity(array1, array2)
        t2 = time.time()
        error = abs(jaccard_similarity1-jaccard_similarity2)
        
        batch_error.append(error)
        batch_time.append(t2-t1)
        batch_jaccard_similarity1.append(jaccard_similarity1)
        batch_jaccard_similarity2.append(jaccard_similarity2)
        
    return batch_error, batch_time, batch_jaccard_similarity1,batch_jaccard_similarity2,array1,array2

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

        batch_error_a, batch_time_a, batch_jaccard_similarity1_a,batch_jaccard_similarity2_a,_,_ = get_feature_deletion_results(Input_dimension = N,Output_dimension = M, default_maps=maps,array1=norm_arr_1,array2=norm_arr_2,mapping_scheme=5,max_value=alpha)

        batch_error_b, batch_time_b,batch_jaccard_similarity1_b,batch_jaccard_similarity2_b,_,_ = get_feature_deletion_results(Input_dimension = N,Output_dimension = M, default_maps=maps,array1=norm_arr_1,array2=norm_arr_2,mapping_scheme=6,max_value=alpha)

        batch_error_c, batch_time_c, batch_jaccard_similarity1_c,batch_jaccard_similarity2_c,_,_ = get_remap_results(Input_dimension = N,Output_dimension = M,array1=norm_arr_1,array2=norm_arr_2,mapping_scheme=5)
        
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

        count += 1

        if count%100 == 0 or count == n_pairs-2:
            np.save('Outputs/sample00020_real_'+dataset+'_'+str(count)+'_'+'.npy', [ np.array(avg_batch_error_a)/count, np.array(avg_batch_error_b)/count, np.array(avg_batch_error_c)/count ])

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
        "NYtimes" : "Data/docword.nytimes.txt",
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
    

    ax[0][0].set(xlabel='% of features deleted', ylabel='MSE')
    ax[0][1].set(xlabel='% of features deleted', ylabel='MSE')
    ax[0][2].set(xlabel='% of features deleted', ylabel='MSE')

    ax[1][0].set(xlabel='% of features deleted', ylabel='Time(s)')
    ax[1][1].set(xlabel='% of features deleted', ylabel='Time(s)')
    ax[1][2].set(xlabel='% of features deleted', ylabel='Time(s)')

    #loop
    it = 0
    for x, y in files.items():
        avg_batch_error_a, avg_batch_error_b, avg_batch_error_c, avg_batch_time_a, avg_batch_time_b, avg_batch_time_c = get_all_errors(y, n_pairs, 5, 6)

        ax[0][it].plot(range(len(avg_batch_error_a)), np.array(avg_batch_error_a)**2, label="No Compensation", linestyle='--')
        ax[0][it].plot(range(len(avg_batch_error_b)), np.array(avg_batch_error_b)**2, label="Our Method", linewidth=3)
        ax[0][it].plot(range(len(avg_batch_error_c)), np.array(avg_batch_error_c)**2, label="Remap")
        ax[0][it].legend(loc='upper right')

        ax[1][it].plot(range(len(avg_batch_time_a)), avg_batch_time_a, label="No Compensation", linestyle='--')
        ax[1][it].plot(range(len(avg_batch_time_b)), avg_batch_time_b, label="Our Method", linewidth=3)
        ax[1][it].plot(range(len(avg_batch_time_c)), avg_batch_time_c, label="Remap")
        ax[1][it].legend(loc='upper right')
        it+=1

    fig.tight_layout(pad=0.5)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    
    #plt.show()
    fig.savefig('Plots/sample_00020_All_Datasets_1000_'+'.png', orientation = 'landscape')

if __name__ == '__main__':
	main()
