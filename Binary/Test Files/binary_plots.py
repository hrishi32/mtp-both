import random
#import matplotlib.pyplot as plt
import sys
from os.path import abspath, exists

# import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from Object_Files.basic_operator import operator
from Object_Files.mapper5 import mapper

# matplotlib.use('agg')

# mapping = mapper(28102,2000)
# default_bits = mapping.bits
# default_maps = mapping.map

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

def load_data(file="Data/docword.enron.txt"):
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

                    if counter > 100 :
                        break



						
                else:
                    break


    # for i in range(105):
    #     data_array.append(np.random.randint(2,size=10000))
    return data_array

    # f_path = abspath(file)
    # if exists(f_path):
    #     with open(f_path) as f:
    #         datapoints = int(f.readline())
    #         features = int(f.readline())
    #         unique_words = int(f.readline())
    #         last_num = 1
    #         feature_array = np.zeros(features,dtype=int)
    #         count = 0
            

def get_feature_deletion_results(Input_dimension ,Output_dimension, default_maps ,array1,array2,mapping_scheme=1,max_value=0):

    batch_error = []
    sample_size = Input_dimension/100
    reduced_input_dim = Input_dimension//2
    demo_operator = operator(input_dim=Input_dimension, output_dim=Output_dimension, mapping_scheme=mapping_scheme)
    # demo_operator.mapping.bits = default_bits
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
        inner_product1, _ = demo_operator.inner_product(array1, array2)
        output1, output2 = demo_operator.get_output_array(array1), demo_operator.get_output_array(array2)
        inner_product2 = inner_product(output1, output2)
        error = abs(inner_product1-inner_product2)
        # print ("inners products:",inner_product1,inner_product2)
        # print("error:", error)
        batch_error.append(error)
        batch_inner_product1.append(inner_product1)
        batch_inner_product2.append(inner_product2)
        # print ("Mapping scheme :",mapping_scheme,"::")
        # print (demo_operator.get_feature_count())
		

    return batch_error,batch_inner_product1,batch_inner_product2,array1,array2

def get_feature_deletion_results_hamming(Input_dimension ,Output_dimension, default_maps ,array1,array2,mapping_scheme=1,max_value=0):

    batch_error = []
    sample_size = Input_dimension/100
    reduced_input_dim = Input_dimension//2
    demo_operator = operator(input_dim=Input_dimension, output_dim=Output_dimension, mapping_scheme=mapping_scheme)
    # demo_operator.mapping.bits = default_bits
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
        hamming_distance1 = original_hamming_distance(array1, array2)
        output1, output2 = demo_operator.get_output_array(array1), demo_operator.get_output_array(array2)
        hamming_distance2 = hamming_distance(output1, output2)
        error = abs(hamming_distance1-hamming_distance2)
        # print ("inners products:",inner_product1,inner_product2)
        # print("error:", error)
        batch_error.append(error)
        batch_inner_product1.append(hamming_distance1)
        batch_inner_product2.append(hamming_distance2)
        # print ("Mapping scheme :",mapping_scheme,"::")
        # print (demo_operator.get_feature_count())
		

    return batch_error,batch_inner_product1,batch_inner_product2,array1,array2

def get_feature_deletion_results_jaccard(Input_dimension ,Output_dimension, default_maps ,array1,array2,mapping_scheme=1,max_value=0):

    batch_error = []
    sample_size = Input_dimension/100
    reduced_input_dim = Input_dimension//2
    demo_operator = operator(input_dim=Input_dimension, output_dim=Output_dimension, mapping_scheme=mapping_scheme)
    # demo_operator.mapping.bits = default_bits
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
        hamming_distance1 = original_jaccard_similarity(array1, array2)
        output1, output2 = demo_operator.get_output_array(array1), demo_operator.get_output_array(array2)
        hamming_distance2 = jaccard_similarity(output1, output2)
        error = abs(hamming_distance1-hamming_distance2)
        # print ("inners products:",inner_product1,inner_product2)
        # print("error:", error)
        batch_error.append(error)
        batch_inner_product1.append(hamming_distance1)
        batch_inner_product2.append(hamming_distance2)
        # print ("Mapping scheme :",mapping_scheme,"::")
        # print (demo_operator.get_feature_count())
		

    return batch_error,batch_inner_product1,batch_inner_product2,array1,array2

def get_feature_deletion_results_cosine(Input_dimension ,Output_dimension, default_maps ,array1,array2,mapping_scheme=1,max_value=0):

    batch_error = []
    sample_size = Input_dimension/100
    reduced_input_dim = Input_dimension//2
    demo_operator = operator(input_dim=Input_dimension, output_dim=Output_dimension, mapping_scheme=mapping_scheme)
    # demo_operator.mapping.bits = default_bits
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
        hamming_distance1 = original_cosine_similarity(array1, array2)
        output1, output2 = demo_operator.get_output_array(array1), demo_operator.get_output_array(array2)
        hamming_distance2 = cosine_similarity(output1, output2)
        error = abs(hamming_distance1-hamming_distance2)
        # print ("inners products:",inner_product1,inner_product2)
        # print("error:", error)
        batch_error.append(error)
        batch_inner_product1.append(hamming_distance1)
        batch_inner_product2.append(hamming_distance2)
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


def predict_ones(reduced_ones, size):
    numerator = np.log10(1-(reduced_ones/size))
    denominator = np.log10(1-(1/size))

    return numerator/denominator

def count_ones(arr):
    result = 0

    for i in arr:
        if i == 1:
            result+=1
    
    return result

def inner_product(arr1, arr2):
    reduced_ones1, reduced_ones2 = count_ones(arr1), count_ones(arr2)

    N = arr1.size
    
    mod_a = predict_ones(reduced_ones1, N)
    mod_b = predict_ones(reduced_ones2, N)

    # print("mod_a:", mod_a, "mod_b:", mod_b)

    ip = 0.0

    for i in range(N):
        ip+=(arr1[i]*arr2[i])
    # print("N: ", N, "mod_a:", mod_a, "mod_b:", mod_b)
    v = ((1-(1.0/N))**mod_a) + ((1-(1.0/N))**mod_b) + (ip*1.0/N) -1.0
    if v <= 0:
        print('\n\n\n\n\n\n\n',((1-(1.0/N))**mod_a),'\t', ((1-(1.0/N))**mod_b),'\t', (ip*1.0/N),  '\n\n\n\n\n\n\n\n')
        numerator = np.log10( v+0.0000001 )
    else:
         numerator = np.log10( v )

    denominator = np.log10(1-(1.0/N))

    # print("numerator", numerator, "denominator:", denominator)

    return mod_a + mod_b - (numerator/denominator)

def hamming_distance(arr1, arr2):
    mod_a = count_ones(arr1)
    mod_b = count_ones(arr2)

    ip = inner_product(arr1, arr2)

    return mod_a + mod_b - (2*ip)

def jaccard_similarity(arr1, arr2):
    ip = inner_product(arr1, arr2)
    hd = hamming_distance(arr1, arr2)

    return ip/(hd+ip)

def cosine_similarity(arr1, arr2):
    mod_a = count_ones(arr1)
    mod_b = count_ones(arr2)
    ip = inner_product(arr1, arr2)

    return ip/((mod_a*mod_b)**0.5)

def original_inner_product(arr1, arr2):
    result = 0

    for i in range(arr1.size):
        result+=(arr1[i]*arr2[i])

    return result

def original_hamming_distance(arr1, arr2):
    result = 0

    for i in range(arr1.size):
        result+=(arr1[i]-arr2[i])

    return result

def original_jaccard_similarity(arr1, arr2):
    numerator = inner_product(arr1, arr2)

    demoninator = 0
    for i, j in zip(arr1, arr2):
        demoninator += int(i|j)

    if demoninator == 0:
        return -1

    return numerator/demoninator

def original_cosine_similarity(arr1, arr2):
    numerator = inner_product(arr1, arr2)

    ones_a, ones_b = count_ones(arr1), count_ones(arr2)

    denominator = ones_a*ones_b

    if denominator == 0:
        return -1

    return numerator/denominator

def mean_squared_error(original, predicted):
    result=0
    for i, j in zip(original, predicted):
        result+=((i-j)**2)

    return result





def plot_ips(arr1):
    demo_operator = operator(input_dim=arr1[0].size, output_dim=arr1[0].size//3, mapping_scheme=5)
    ip1, ip2 = np.zeros(arr1.shape[0], dtype=float), np.zeros(arr1.shape[0], dtype=float)
    print("arr1", arr1, "size", arr1.shape)
    for i in range(arr1.shape[0]-1):
        ip1[i], ip2[i] = demo_operator.inner_product(arr1[i], arr1[(i+1)])
    
    output_array = []
    for i in range(arr1.shape[0]):
        output_array.append(demo_operator.get_output_array(arr1[i]))

    for i in range(arr1.shape[0]-1):
        ip2[i] = inner_product(output_array[i], output_array[i+1])

    plt.plot(range(arr1.shape[0]), ip1, label="Original Inner Product")
    plt.plot(range(arr1.shape[0]), ip2, label="Reduced Vector Inner Product")
    plt.legend()

    plt.show()





def main():
    # plt.figure(figsize=(7, 2.5))
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    ax1.set_title('Inner Product')
    ax2.set_title('Hamming Distance')
    ax3.set_title('Jaccard Similarity')
    ax4.set_title('Cosine Similarity')

    ax1.set(xlabel='', ylabel='Error in Similarity Measure')
    ax2.set(xlabel='', ylabel='')
    ax3.set(xlabel='% of Feature Deletion', ylabel='')
    ax4.set(xlabel='', ylabel='')

    n_args = len(sys.argv)
    if n_args >= 2:
        input_file = sys.argv[1]
    else:
        input_file = 'blah'
    compensation1, compensation2 = 0, 1 # 0 = No Compensaation, 1 = 1 step Compensation, 2 = 2 step
    
    if n_args > 2:
        compensation1 = int(sys.argv[2])
        compensation2 = int(sys.argv[3])
    
    m1, m2 = 5, 6 #One without compensation, other with one step compensation
    if compensation1 == 0:
        m1 = 5
    elif compensation1 == 1:
        m1 = 6
    else:
        m1 = 8

    if compensation2 == 0:
        m2 = 5
    elif compensation2 == 1:
        m2 = 6
    else:
        m2 = 8

    n_pairs = 100

    if n_args >3:
        n_pairs = int(sys.argv[4])


    data_array = load_data(input_file)
    N = data_array[0].size
    M = N//10

    binary_operator0 = operator(N, M, 5) # No compensation
    # binary_operator1 = operator(N, M, 6) # 1 step compensation
    # binary_operator2 = operator(N, M, 8) # 2 step compensation

    default_maps = binary_operator0.mapping.map # fixing this mapping to all the scemes
    iterations = 1 #len(data_array)-1
    # Inner Product
    # print("Length of Data array", len(data_array))
    for i in range(iterations):
        print('\n Iteration', i , '\n')
        batch_error_a,_,_,_,_ = get_feature_deletion_results(Input_dimension = N,Output_dimension = M, default_maps=default_maps,array1=data_array[i],array2=data_array[i+1],mapping_scheme=5) # Without Compensation

        batch_error_b,_,_,_,_ = get_feature_deletion_results(Input_dimension = N,Output_dimension = M, default_maps=default_maps,array1=data_array[i],array2=data_array[i+1],mapping_scheme=6) # With 1 step Compensation

        # batch_error_c,_,_,_,_ = get_feature_deletion_results(Input_dimension = N,Output_dimension = M, default_maps=default_maps,array1=data_array[i],array2=data_array[i+1],mapping_scheme=8) # With 2 step Compensation

        if i == 0:
            avg_batch_error_a = batch_error_a
            avg_batch_error_b = batch_error_b
            # avg_batch_error_c = batch_error_c

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
                # avg_batch_error_c[i] += batch_error_c[i]

                # avg_inner_product1_a[i] += batch_inner_product1_a[i]
                # avg_inner_product2_a[i] += batch_inner_product2_a[i]
                # avg_inner_product1_b[i] += batch_inner_product1_b[i]
                # avg_inner_product2_b[i] += batch_inner_product2_b[i]
                # avg_inner_product1_c[i] += batch_inner_product1_c[i]
                # avg_inner_product2_c[i] += batch_inner_product2_c[i]
    
    for i in range(len(avg_batch_error_a)):
        avg_batch_error_a[i] /= (len(data_array)-1)
        avg_batch_error_b[i] /= (len(data_array)-1)
        # avg_batch_error_c[i] /= (len(data_array)-1)

        # avg_inner_product1_a[i] /= (len(data_array)-1)
        # avg_inner_product2_a[i] /= (len(data_array)-1)
        # avg_inner_product1_b[i] /= (len(data_array)-1)
        # avg_inner_product2_b[i] /= (len(data_array)-1)
        # avg_inner_product1_c[i] /= (len(data_array)-1)
        # avg_inner_product2_c[i] /= (len(data_array)-1)

    ax1.plot(range(len(avg_batch_error_a)), avg_batch_error_a, label = "NO Compensation")
    # plt.plot(range(len(avg_inner_product1_a)), avg_inner_product1_a, label = "HD1 With "+str(0)+" step Compensation")
    # plt.plot(range(len(avg_inner_product2_a)), avg_inner_product2_a, label = "HD2 With "+str(0)+" step Compensation")

    ax1.plot(range(len(avg_batch_error_b)), avg_batch_error_b, label = str(1)+" step Compensation")
    # plt.plot(range(len(avg_inner_product1_b)), avg_inner_product1_b, label = "HD1 With "+str(1)+" step Compensation")
    # plt.plot(range(len(avg_inner_product2_b)), avg_inner_product2_b, label = "HD2 With "+str(1)+" step Compensation")

    # ax1.plot(range(len(avg_batch_error_c)), avg_batch_error_c, label = str(2)+" step Compensation")
    # plt.plot(range(len(avg_inner_product1_c)), avg_inner_product1_c, label = "HD1 With "+str(2)+" step Compensation")
    # plt.plot(range(len(avg_inner_product2_c)), avg_inner_product2_c, label = "HD2 With "+str(2)+" step Compensation")

    # plt.title(input_file.split('/')[1].split('.')[1]+' Dataset')
    # plt.xlabel('% of feature deletion')
    # plt.ylabel('MSE, Hamming Distance')



    # Hamming Distance


    # data_array = load_data(input_file)

    print(len(data_array))
    for i in range(iterations):
        print('\n Iteration Hamming', i , '\n')
        batch_error_a,_,_,_,_ = get_feature_deletion_results_hamming(Input_dimension = N,Output_dimension = M, default_maps=default_maps,array1=data_array[i],array2=data_array[i+1],mapping_scheme=5) # Without Compensation

        batch_error_b,_,_,_,_ = get_feature_deletion_results_hamming(Input_dimension = N,Output_dimension = M, default_maps=default_maps,array1=data_array[i],array2=data_array[i+1],mapping_scheme=6) # With 1 step Compensation

        # batch_error_c,_,_,_,_ = get_feature_deletion_results_hamming(Input_dimension = N,Output_dimension = M, default_maps=default_maps,array1=data_array[i],array2=data_array[i+1],mapping_scheme=8) # With 2 step Compensation

        if i == 0:
            avg_batch_error_a = batch_error_a
            avg_batch_error_b = batch_error_b
            # avg_batch_error_c = batch_error_c

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
                # avg_batch_error_c[i] += batch_error_c[i]

                # avg_inner_product1_a[i] += batch_inner_product1_a[i]
                # avg_inner_product2_a[i] += batch_inner_product2_a[i]
                # avg_inner_product1_b[i] += batch_inner_product1_b[i]
                # avg_inner_product2_b[i] += batch_inner_product2_b[i]
                # avg_inner_product1_c[i] += batch_inner_product1_c[i]
                # avg_inner_product2_c[i] += batch_inner_product2_c[i]
    
    for i in range(len(avg_batch_error_a)):
        avg_batch_error_a[i] /= (len(data_array)-1)
        avg_batch_error_b[i] /= (len(data_array)-1)
        # avg_batch_error_c[i] /= (len(data_array)-1)

        # avg_inner_product1_a[i] /= (len(data_array)-1)
        # avg_inner_product2_a[i] /= (len(data_array)-1)
        # avg_inner_product1_b[i] /= (len(data_array)-1)
        # avg_inner_product2_b[i] /= (len(data_array)-1)
        # avg_inner_product1_c[i] /= (len(data_array)-1)
        # avg_inner_product2_c[i] /= (len(data_array)-1)


    ax2.plot(range(len(avg_batch_error_a)), avg_batch_error_a)#, label = "Error With "+str(0)+" step Compensation")
    # plt.plot(range(len(avg_inner_product1_a)), avg_inner_product1_a, label = "HD1 With "+str(0)+" step Compensation")
    # plt.plot(range(len(avg_inner_product2_a)), avg_inner_product2_a, label = "HD2 With "+str(0)+" step Compensation")

    ax2.plot(range(len(avg_batch_error_b)), avg_batch_error_b)#, label = "Error With "+str(1)+" step Compensation")
    # plt.plot(range(len(avg_inner_product1_b)), avg_inner_product1_b, label = "HD1 With "+str(1)+" step Compensation")
    # plt.plot(range(len(avg_inner_product2_b)), avg_inner_product2_b, label = "HD2 With "+str(1)+" step Compensation")

    # ax2.plot(range(len(avg_batch_error_c)), avg_batch_error_c)#, label = "Error With "+str(2)+" step Compensation")
    # plt.plot(range(len(avg_inner_product1_c)), avg_inner_product1_c, label = "HD1 With "+str(2)+" step Compensation")
    # plt.plot(range(len(avg_inner_product2_c)), avg_inner_product2_c, label = "HD2 With "+str(2)+" step Compensation")

    # plt.title(input_file.split('/')[1].split('.')[1]+' Dataset')
    # plt.xlabel('% of feature deletion')
    # plt.ylabel('MSE, Hamming Distance')


#########################################################################################
















    # Jaccard Similarity

    # data_array = load_data(input_file)

    # print("Length of Data Array", len(data_array))
    for i in range(iterations):
        print('\n Iteratioon Jaccard', i , '\n')
        batch_error_a,__,__,_,_ = get_feature_deletion_results_jaccard(Input_dimension = N,Output_dimension = M, default_maps=default_maps,array1=data_array[i],array2=data_array[i+1],mapping_scheme=5) # Without Compensation

        batch_error_b,__,__,_,_ = get_feature_deletion_results_jaccard(Input_dimension = N,Output_dimension = M, default_maps=default_maps,array1=data_array[i],array2=data_array[i+1],mapping_scheme=6) # With 1 step Compensation

        # batch_error_c,__,__,_,_ = get_feature_deletion_results_jaccard(Input_dimension = N,Output_dimension = M, default_maps=default_maps,array1=data_array[i],array2=data_array[i+1],mapping_scheme=8) # With 2 step Compensation

        if i == 0:
            avg_batch_error_a = batch_error_a
            avg_batch_error_b = batch_error_b
            # avg_batch_error_c = batch_error_c

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
                # avg_batch_error_c[i] += batch_error_c[i]

                # avg_inner_product1_a[i] += batch_inner_product1_a[i]
                # avg_inner_product2_a[i] += batch_inner_product2_a[i]
                # avg_inner_product1_b[i] += batch_inner_product1_b[i]
                # avg_inner_product2_b[i] += batch_inner_product2_b[i]
                # avg_inner_product1_c[i] += batch_inner_product1_c[i]
                # avg_inner_product2_c[i] += batch_inner_product2_c[i]
    
    for i in range(len(avg_batch_error_a)):
        avg_batch_error_a[i] /= (len(data_array)-1)
        avg_batch_error_b[i] /= (len(data_array)-1)
        # avg_batch_error_c[i] /= (len(data_array)-1)

        # avg_inner_product1_a[i] /= (len(data_array)-1)
        # avg_inner_product2_a[i] /= (len(data_array)-1)
        # avg_inner_product1_b[i] /= (len(data_array)-1)
        # avg_inner_product2_b[i] /= (len(data_array)-1)
        # avg_inner_product1_c[i] /= (len(data_array)-1)
        # avg_inner_product2_c[i] /= (len(data_array)-1)


    ax3.plot(range(len(avg_batch_error_a)), avg_batch_error_a)#, label = "Error With "+str(0)+" step Compensation")
    # plt.plot(range(len(avg_inner_product1_a)), avg_inner_product1_a, label = "HD1 With "+str(0)+" step Compensation")
    # plt.plot(range(len(avg_inner_product2_a)), avg_inner_product2_a, label = "HD2 With "+str(0)+" step Compensation")

    ax3.plot(range(len(avg_batch_error_b)), avg_batch_error_b)#, label = "Error With "+str(1)+" step Compensation")
    # plt.plot(range(len(avg_inner_product1_b)), avg_inner_product1_b, label = "HD1 With "+str(1)+" step Compensation")
    # plt.plot(range(len(avg_inner_product2_b)), avg_inner_product2_b, label = "HD2 With "+str(1)+" step Compensation")

    # ax3.plot(range(len(avg_batch_error_c)), avg_batch_error_c)#, label = "Error With "+str(2)+" step Compensation")
    # plt.plot(range(len(avg_inner_product1_c)), avg_inner_product1_c, label = "HD1 With "+str(2)+" step Compensation")
    # plt.plot(range(len(avg_inner_product2_c)), avg_inner_product2_c, label = "HD2 With "+str(2)+" step Compensation")

    # plt.title(input_file.split('/')[1].split('.')[1]+' Dataset')
    # plt.xlabel('% of feature deletion')
    # plt.ylabel('MSE, Hamming Distance')


#############################################################################################


# Cosine Similarity

    # data_array = load_data(input_file)

    # print("Length of Data Array", len(data_array))
    for i in range(iterations):
        print('\n Iteration Cosine', i , '\n')
        batch_error_a,_,_,_,_ = get_feature_deletion_results_cosine(Input_dimension = N,Output_dimension = M, default_maps=default_maps,array1=data_array[i],array2=data_array[i+1],mapping_scheme=5) # Without Compensation

        batch_error_b,_,_,_,_ = get_feature_deletion_results_cosine(Input_dimension = N,Output_dimension = M, default_maps=default_maps,array1=data_array[i],array2=data_array[i+1],mapping_scheme=6) # With 1 step Compensation

        # batch_error_c,_,_,_,_ = get_feature_deletion_results_cosine(Input_dimension = N,Output_dimension = M, default_maps=default_maps,array1=data_array[i],array2=data_array[i+1],mapping_scheme=8) # With 2 step Compensation

        if i == 0:
            avg_batch_error_a = batch_error_a
            avg_batch_error_b = batch_error_b
            # avg_batch_error_c = batch_error_c

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
                # avg_batch_error_c[i] += batch_error_c[i]

                # avg_inner_product1_a[i] += batch_inner_product1_a[i]
                # avg_inner_product2_a[i] += batch_inner_product2_a[i]
                # avg_inner_product1_b[i] += batch_inner_product1_b[i]
                # avg_inner_product2_b[i] += batch_inner_product2_b[i]
                # avg_inner_product1_c[i] += batch_inner_product1_c[i]
                # avg_inner_product2_c[i] += batch_inner_product2_c[i]
    
    for i in range(len(avg_batch_error_a)):
        avg_batch_error_a[i] /= (len(data_array)-1)
        avg_batch_error_b[i] /= (len(data_array)-1)
        # avg_batch_error_c[i] /= (len(data_array)-1)

        # avg_inner_product1_a[i] /= (len(data_array)-1)
        # avg_inner_product2_a[i] /= (len(data_array)-1)
        # avg_inner_product1_b[i] /= (len(data_array)-1)
        # avg_inner_product2_b[i] /= (len(data_array)-1)
        # avg_inner_product1_c[i] /= (len(data_array)-1)
        # avg_inner_product2_c[i] /= (len(data_array)-1)


    ax4.plot(range(len(avg_batch_error_a)), avg_batch_error_a)#, label = "Error With "+str(0)+" step Compensation")
    # plt.plot(range(len(avg_inner_product1_a)), avg_inner_product1_a, label = "HD1 With "+str(0)+" step Compensation")
    # plt.plot(range(len(avg_inner_product2_a)), avg_inner_product2_a, label = "HD2 With "+str(0)+" step Compensation")

    ax4.plot(range(len(avg_batch_error_b)), avg_batch_error_b)#, label = "Error With "+str(1)+" step Compensation")
    # plt.plot(range(len(avg_inner_product1_b)), avg_inner_product1_b, label = "HD1 With "+str(1)+" step Compensation")
    # plt.plot(range(len(avg_inner_product2_b)), avg_inner_product2_b, label = "HD2 With "+str(1)+" step Compensation")

    # ax4.plot(range(len(avg_batch_error_c)), avg_batch_error_c)#, label = "Error With "+str(2)+" step Compensation")
    # plt.plot(range(len(avg_inner_product1_c)), avg_inner_product1_c, label = "HD1 With "+str(2)+" step Compensation")
    # plt.plot(range(len(avg_inner_product2_c)), avg_inner_product2_c, label = "HD2 With "+str(2)+" step Compensation")

    # plt.title(input_file.split('/')[1].split('.')[1]+' Dataset')
    # plt.xlabel('% of feature deletion')
    # plt.ylabel('MSE, Hamming Distance')















    # ax1.label_outer()
    # ax2.label_outer()
    # ax3.label_outer()
    # ax4.label_outer()
    fig.legend()
    fig.set_figheight(5)
    fig.set_figwidth(17)
    #plt.show()
    fig.savefig('Plots/'+input_file.split('/')[-1]+'_all_plots.png', orientation = 'landscape')
    # plt.show()

    # binary_operator1.mapping.map = default_maps
    # binary_operator2.mapping.map = default_maps


    # output_array = []
    # inner_product_array = []
    # hamming_distance_array = []

    # for i in data_array:
    #     output_array.append(binary_operator0.get_output_array(i))

    # output_array = np.array(output_array)


        
    # count = 1
    # avg_batch_error_a = []
    # avg_batch_error_b = []
    # avg_inner_product1_a = []
    # avg_inner_product2_a = []
    # avg_inner_product1_b = []
    # avg_inner_product2_b = []
    # data_array = load_data(input_file)
    # N = data_array[0].size
    # M = 2000

    # plot_ips(np.array(data_array))

    # exit()
  
    
    # while count < n_pairs:
    #     print(count)
        
    #     # print ("* Input Dimension of Dataset:",N)
    #     # print ("* Output (compressed) Dimension of Dataset:",M)
    #     alpha = 1

    #     arr1 = data_array[count-1]
    #     arr2 = data_array[count]

    #     # print ("* Selected array (1) from Dataset:",arr1)
    #     # print ("* Selected array (2) from Dataset:",arr2)

    #     # norm_arr_1 = array_normalization(arr1)
    #     # norm_arr_2 = array_normalization(arr2)

    #     norm_arr_1 = arr1
    #     norm_arr_2 = arr2

    #     # print ("* Normalized array (1):",norm_arr_1)
    #     # print ("* Normalized array (2):",norm_arr_2)

    #     batch_error_a,batch_inner_product1_a,batch_inner_product2_a,_,_ = get_feature_deletion_results(Input_dimension = N,Output_dimension = M,array1=norm_arr_1,array2=norm_arr_2,mapping_scheme=m1,max_value=alpha)

    #     # plt.plot(range(len(batch_error)), batch_error, label = "Error Without Compensation")
    #     # plt.plot(range(len(batch_inner_product1)), batch_inner_product1, label = "IP1 Without Compensation")
    #     # plt.plot(range(len(batch_inner_product2)), batch_inner_product2, label = "IP2 Without Compensation")

    #     batch_error_b,batch_inner_product1_b,batch_inner_product2_b,_,_ = get_feature_deletion_results(Input_dimension = N,Output_dimension = M,array1=norm_arr_1,array2=norm_arr_2,mapping_scheme=m2,max_value=alpha)

    #     # print(batch_error,batch_inner_product1,batch_inner_product2,array1,array2)

    #     # plt.plot(range(len(batch_error)), batch_error, label = "Error With Compensation")
    #     # plt.plot(range(len(batch_inner_product1)), batch_inner_product1, label = "IP1 With Compensation")
    #     # plt.plot(range(len(batch_inner_product2)), batch_inner_product2, label = "IP2 With Compensation")
    #     # plt.legend()
    #     # plt.show()
    #     if count == 1:
    #         avg_batch_error_a = batch_error_a
    #         avg_batch_error_b = batch_error_b
    #         avg_inner_product1_a = batch_inner_product1_a
    #         avg_inner_product2_a = batch_inner_product2_a
    #         avg_inner_product1_b = batch_inner_product1_b
    #         avg_inner_product2_b = batch_inner_product2_b

    #     else :
    #         for i in range(len(batch_error_a)):
    #             avg_batch_error_a[i] += batch_error_a[i]
    #             avg_batch_error_b[i] += batch_error_b[i]
    #             avg_inner_product1_a[i] += batch_inner_product1_a[i]
    #             avg_inner_product2_a[i] += batch_inner_product2_a[i]
    #             avg_inner_product1_b[i] += batch_inner_product1_b[i]
    #             avg_inner_product2_b[i] += batch_inner_product2_b[i]
    #     count += 1

    # for i in range(len(avg_batch_error_a)):
    #     avg_batch_error_a[i] /= 100
    #     avg_batch_error_b[i] /= 100
    #     avg_inner_product1_a[i] /= 100
    #     avg_inner_product2_a[i] /= 100
    #     avg_inner_product1_b[i] /= 100
    #     avg_inner_product2_b[i] /= 100

    # plt.plot(range(len(avg_batch_error_a)), avg_batch_error_a, label = "Error With "+str(compensation1)+" step Compensation")
    # plt.plot(range(len(avg_inner_product1_a)), avg_inner_product1_a, label = "IP1 With "+str(compensation1)+" step Compensation")
    # plt.plot(range(len(avg_inner_product2_a)), avg_inner_product2_a, label = "IP2 With "+str(compensation1)+" step Compensation")

    # plt.plot(range(len(avg_batch_error_b)), avg_batch_error_b, label = "Error With "+str(compensation2)+" step Compensation")
    # plt.plot(range(len(avg_inner_product1_b)), avg_inner_product1_b, label = "IP1 With "+str(compensation2)+" step Compensation")
    # plt.plot(range(len(avg_inner_product2_b)), avg_inner_product2_b, label = "IP2 With "+str(compensation2)+" step Compensation")
    # plt.legend()

    # #plt.show()
    # plt.savefig('Plots/'+input_file.split('/')[-1]+'_plot_'+str(compensation1)+'_'+str(compensation2)+'.png')



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
