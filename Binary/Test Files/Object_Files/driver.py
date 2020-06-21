# from mapper import mapper
import numpy as np
from mapper2 import mapper as mapper2
from mapper3 import mapper as mapper3
from mapper4 import mapper as mapper4
from mapper5 import mapper as mapper5
from mapper6 import mapper as mapper6
from mapper7 import mapper as mapper7
from mapper8 import mapper as mapper8
from basic_operator import operator
import matplotlib.pyplot as plt
from os.path import abspath, exists
import sys
import time

# import random

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
    # print("num/denom", numerator/denominator)

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
        result+=abs(arr1[i]-arr2[i])

    return result

def original_jaccard_similarity(arr1, arr2):
    numerator = original_inner_product(arr1, arr2)

    demoninator = 0
    for i, j in zip(arr1, arr2):
        demoninator += int(i|j)

    if demoninator == 0:
        return -1

    return numerator/demoninator

def original_cosine_similarity(arr1, arr2):
    # numerator = original_inner_product(arr1, arr2)

    # ones_a, ones_b = count_ones(arr1), count_ones(arr2)

    # denominator = ones_a*ones_b

    # if denominator == 0:
    #     return -1

    return np.dot(arr1, arr2)/(np.linalg.norm(arr1)*np.linalg.norm(arr2))

def mean_squared_error(original, predicted):
    result=0
    for i, j in zip(original, predicted):
        result+=((i-j)**2)

    return result

def read_data(file=''):
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

def process(array1, features1, spar, oper):
    t1 = time.time()
    array1 = np.concatenate((array1, features1), axis=0)
    # array2 = np.concatenate((array2, features2), axis=0)

    features_sparsity = sparsity(features1)
    # feature2_sparsity = 0

    # for i in range(features1.size):
    #     if features1[i] == 1:
    #         feature1_sparsity += 1
        # if features2[i] == 1:
        #     feature2_sparsity += 1

    # features_sparsity = 0
    # if feature1_sparsity > feature2_sparsity:
    #     features_sparsity = feature1_sparsity
    # else:
    #     features_sparsity = feature2_sparsity

    out_size = (spar + features_sparsity)**2

    t2 = time.time()
    # old_output_size = self.mapping.output_dim()
    bins = abs(oper.mapping.output_dimension - out_size)
    # print("Bins:", bins)



    oper.mapping.new_batch_insert_feature(features1.size, bins, features_sparsity>0)

    t3 = time.time()

    scheme_time = t3-t1

    oper2 = operator(array1.size, out_size, 8)

    t4 = time.time()

    remap_time = (t4-t3) + (t2-t1)

    # if scheme_time > remap_time:
    # print(bins)

    return scheme_time, remap_time

def sparsity(arr):
    count = 0
    for i in arr:
        count+=i

    return count

def mse(lst1, lst2):
    return abs(np.array(lst1) - np.array(lst2))

def brute_vs_scheme(data):
    ip_b = []
    ip_s = []

    hd_b = []
    hd_s = []

    js_b = []
    js_s = []

    cs_b = []
    cs_s = []


    for i in range(len(data)-1):
        s = max(sparsity(data[i]), sparsity(data[i+1]))
        oper = operator(data[i].size, s**2, mapping_scheme=8)

        print(i)

        insertion_result = oper.new_insert_feature(data[i], data[i+1], data[i+1][500:1000], data[i][500:1000], max(sparsity(data[i]), sparsity(data[i+1])))

        ip_b.append(insertion_result[0])
        ip_s.append(insertion_result[1])
        hd_b.append(insertion_result[2])
        hd_s.append(insertion_result[3])
        js_b.append(insertion_result[4])
        js_s.append(insertion_result[5])
        cs_b.append(insertion_result[6])
        cs_s.append(insertion_result[7])

    return (ip_b, ip_s), (hd_b, hd_s), (js_b, js_s), (cs_b, cs_s)


    
    

def main():
    data = read_data(file="../Data/docword.nytimes.txt")

    # scheme_time = []
    # remap_time = []
    
    # for i in range(len(data)-1):
    #     print(i)
    #     s = sparsity(data[i])
    #     oper = operator(data[i].size, s**2, mapping_scheme=8)

    #     a, b = process(data[i], data[i][:1000], sparsity(data[i]), oper)

    #     scheme_time.append(a)
    #     remap_time.append(b)
        
    # # print(data)

    # plt.title("enron Dataset")
    # plt.xlabel("# iterations")
    # plt.ylabel("Time taken for insertion")

    # plt.plot(range(len(scheme_time)), scheme_time, label="Scheme time")
    # plt.plot(range(len(remap_time)), remap_time, label="Remap time")
    # plt.legend()

    # plt.savefig("../Plots/time_insertion_enron1.png")
    
    # return data

    ip, hd, js, cs = brute_vs_scheme(data)
    number_of_features = range(len(ip[0]))

    fig, ax = plt.subplots(2, 2)
    
    ax[0][0].set_title('Inner Product')
    ax[0][1].set_title('Hamming Distance')
    ax[1][0].set_title('Jaccard Similarity')
    ax[1][1].set_title('Cosine Similarity')

    ax[0][0].set(xlabel='# features inserted', ylabel='Error in Similarity Measure')
    ax[0][1].set(xlabel='# features inserted', ylabel='Error in Similarity Measure')
    ax[1][0].set(xlabel='# features inserted', ylabel='Error in Similarity Measure')
    ax[1][1].set(xlabel='# features inserted', ylabel='Error in Similarity Measure')

    ax[0][0].plot(number_of_features, mse(ip[0], ip[1]), label="Error")
    ax[0][0].plot(number_of_features, ip[0], label="Measure: Brute Force")
    ax[0][0].plot(number_of_features, ip[1], label="Measure: Scheme")

    ax[0][1].plot(number_of_features, mse(hd[0], hd[1]))
    ax[0][1].plot(number_of_features, hd[0])
    ax[0][1].plot(number_of_features, hd[1])

    ax[1][0].plot(number_of_features, mse(js[0], js[1]))
    ax[1][0].plot(number_of_features, js[0])
    ax[1][0].plot(number_of_features, js[1])

    ax[1][1].plot(number_of_features, mse(cs[0], cs[1]))
    ax[1][1].plot(number_of_features, cs[0])
    ax[1][1].plot(number_of_features, cs[1])


    fig.legend()
    fig.tight_layout(pad=0.5)
    fig.set_figheight(9)
    fig.set_figwidth(9)

    
    #plt.show()
    fig.savefig('../Plots/insertion_measures_nytimes.png', orientation = 'landscape')




if __name__ == "__main__":
    main()