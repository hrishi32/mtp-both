from mapper import mapper
import numpy as np
from mapper2 import mapper as mapper2
from mapper3 import mapper as mapper3
from mapper4 import mapper as mapper4
from mapper5 import mapper as mapper5
from mapper6 import mapper as mapper6
from mapper7 import mapper as mapper7
from mapper8 import mapper as mapper8
import matplotlib.pyplot as plt

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

class operator:
    def __init__(self, input_dim=50, output_dim = 15,mapping_scheme=1):
        if mapping_scheme == 1:
            self.mapping = mapper(input_dim=input_dim, out_dim=output_dim)
        elif mapping_scheme == 2:
            self.mapping = mapper2(input_dim=input_dim, out_dim=output_dim)
        elif mapping_scheme == 3:
            self.mapping = mapper3(input_dim=input_dim, out_dim=output_dim)
        elif mapping_scheme == 4:
            self.mapping = mapper4(input_dim=input_dim, out_dim=output_dim)
        elif mapping_scheme == 5:
            self.mapping = mapper5(input_dim=input_dim, out_dim=output_dim)
        elif mapping_scheme == 6:
            self.mapping = mapper6(input_dim=input_dim, out_dim=output_dim)
        elif mapping_scheme == 7:
            self.mapping = mapper7(input_dim=input_dim, out_dim=output_dim)
        elif mapping_scheme == 8:
            self.mapping = mapper8(input_dim=input_dim, out_dim=output_dim)  


    def insert_feature(self, position=0, array1 = [], array2 = [], value1 = 0, value2 = 0):
        self.mapping.insert_feature(position=position)
        array1 = np.insert(array1, position, value1)
        array2 = np.insert(array2, position, value2)
        return array1,array2

    def delete_feature(self, position=0, array1 = [], array2 = []):
        self.mapping.delete_feature(position=position)
        array1 = np.delete(array1, position)
        array2 = np.delete(array2, position)
        return array1, array2

    def batch_insert_feature(self,batch_positions=[],array1=[],array2=[],batch_value1=[],batch_value2=[]):
        
        flags = []
        for i in range(self.mapping.input_dimension):
            flags.append([])

        for i in range(len(batch_positions)):
            flags[batch_positions[i]] = [batch_value1[i],batch_value2[i]]


        i = 0
        factor = 0
        old_dim = self.mapping.input_dimension
        last_insertion = 0
        # print ("start")
        while i < old_dim:

            # print (i,flags[i])
            if len(flags[i]) != 0 and last_insertion == 0 :
                array1 = np.insert(array1,i+factor,flags[i][0])
                array2 = np.insert(array2,i+factor,flags[i][0])
                factor+=1
                last_insertion +=1
                # flags = np.insert(flags, i, 0)
            elif len(flags[i]) != 0:
                array1 = np.insert(array1,i+factor-last_insertion,flags[i][0])
                array2 = np.insert(array2,i+factor-last_insertion,flags[i][0])
                # self.insert_feature(i+factor-last_insertion)
                factor+=1
                last_insertion+=1
            else:
                last_insertion = 0
            
            i+=1
        
        #     # self.insert_feature(position=batch_positions[i])

        # i = 0
        # # print ("start")
        # while i < self.input_dimension:

        #     print (i,flags[i])
        #     if len(flags[i])!=0:
        #         flags.insert(i, [])
        #         array1 = np.insert(array1, i, flags[i][0] )
        #         array2 = np.insert(array2, i, flags[i][1])

    
        #         i += 1
        #     i+=1
        self.mapping.batch_insert_feature(batch_positions=batch_positions)

        
        return array1,array2

    def batch_delete_feature(self,batch_positions=[],array1=[],array2=[]):

        flags = np.zeros(self.mapping.input_dimension)
        for i in range(len(batch_positions)):
            flags[batch_positions[i]] = 1

        i = 0
        factor = 0
        old_dim = self.mapping.input_dimension
        last_deletion = 0
        # print ("start")
        while i < old_dim:

            # print (i,flags[i])
            if flags[i] == 1 and last_deletion == 0 :
                array1 = np.delete(array1,i-factor)
                array2 = np.delete(array2,i-factor)
                factor+=1
                last_deletion +=1
                # flags = np.insert(flags, i, 0)
            elif flags[i] == 1:
                array1 = np.delete(array1,i-factor)
                array2 = np.delete(array2,i-factor)
                # self.insert_feature(i+factor-last_insertion)
                factor+=1
                last_deletion+=1
            else:
                last_deletion = 0
            
            i+=1
        
        
        self.mapping.batch_delete_feature(batch_positions=batch_positions)
        # print(self.get_feature_count())

        
        return array1,array2

    def array_normalization(self, input_array):
        # array_norm = np.linalg.norm(input_array)
        # # print ("array norm:",array_norm)
        # result = np.zeros(input_array.size, dtype=float)
        # for i in range(input_array.size):
        #     result[i] = (1.0*input_array[i])/array_norm

        return input_array

    # def inner_product(self, input_array1, input_array2):
    #     input_array1 = self.array_normalization(input_array1)
    #     input_array2 = self.array_normalization(input_array2)

    #     # print ("norm array1 :",input_array1)
    #     # print ("norm array2 :",input_array2)

    #     output_array1 = self.mapping.dimension_reduction(input_array1)
    #     output_array2 = self.mapping.dimension_reduction(input_array2)

    #     #print("Output1", output_array1)
    #     #print("Output2", output_array2)

    #     result1, result2 = 0, 0
        
    #     for i, j in zip(input_array1, input_array2):
    #         result1+=(i*j)

    #     for i, j in zip(output_array1, output_array2):
    #         result2+=(i*j)

    #     #print("Input Inner Product:", result1)
    #     #print("Output Inner Product:", result2)

    #     return result1, result2*2

    def get_output_array(self, arr):
        return self.mapping.dimension_reduction(arr)


    def get_feature_counter(self):
        return self.mapping.get_feature_counter()

    def get_feature_count(self):
        return self.mapping.get_feature_count()




    #**************************************************************************#

    

    def new_insert_feature(self, array1, array2, features1, features2, sparsity):

        # comp_arr1, comp_arr2 = self.mapping.dimension_reduction(array1), self.mapping.dimension_reduction(array2)
        # before_oip = original_inner_product(array1, array2)
        # before_ip = inner_product(comp_arr1, comp_arr2)

        # b_ohd = original_hamming_distance(array1, array2)
        # b_hd = hamming_distance(comp_arr1, comp_arr2)

        # b_ojs = original_jaccard_similarity(array1, array2)
        # b_js = jaccard_similarity(comp_arr1, comp_arr2)

        # b_ocs = original_cosine_similarity(array1, array2)
        # b_cs = cosine_similarity(comp_arr1, comp_arr2)

        # print("Original ip", before_oip)
        # print("ip", before_ip)

        # print("original hd", original_hamming_distance(array1, array2))
        # print("hd", hamming_distance(self.mapping.dimension_reduction(array1), self.mapping.dimension_reduction(array2)))

        # print(original_jaccard_similarity(array1, array2))
        # print(jaccard_similarity(self.mapping.dimension_reduction(array1), self.mapping.dimension_reduction(array2)))

        # print(original_cosine_similarity(array1, array2))
        # print(cosine_similarity(self.mapping.dimension_reduction(array1), self.mapping.dimension_reduction(array2)))
        array1 = np.concatenate((array1, features1), axis=0)
        array2 = np.concatenate((array2, features2), axis=0)

        feature1_sparsity = 0
        feature2_sparsity = 0

        for i in range(features1.size):
            if features1[i] == 1:
                feature1_sparsity += 1
            if features2[i] == 1:
                feature2_sparsity += 1

        features_sparsity = 0
        if feature1_sparsity > feature2_sparsity:
            features_sparsity = feature1_sparsity
        else:
            features_sparsity = feature2_sparsity

        out_size = (sparsity + features_sparsity)**2
        # old_output_size = self.mapping.output_dim()
        bins = abs(self.mapping.output_dimension - out_size)



        self.mapping.new_batch_insert_feature(features1.size, bins, features_sparsity>0)

        # print("Original ip",original_inner_product(array1, array2))
        # print("ip", inner_product(self.mapping.dimension_reduction(array1), self.mapping.dimension_reduction(array2)))

        comp_arr1, comp_arr2 = self.mapping.dimension_reduction(array1), self.mapping.dimension_reduction(array2)

        after_oip = original_inner_product(array1, array2)
        after_ip = inner_product(comp_arr1, comp_arr2)

        ohd = original_hamming_distance(array1, array2)
        hd = hamming_distance(comp_arr1, comp_arr2)

        ojs = original_jaccard_similarity(array1, array2)
        js = jaccard_similarity(comp_arr1, comp_arr2)

        ocs = original_cosine_similarity(array1, array2)
        cs = cosine_similarity(comp_arr1, comp_arr2)


        # print(original_hamming_distance(array1, array2))
        # print(hamming_distance(self.mapping.dimension_reduction(array1), self.mapping.dimension_reduction(array2)))

        # print(original_jaccard_similarity(array1, array2))
        # print(jaccard_similarity(self.mapping.dimension_reduction(array1), self.mapping.dimension_reduction(array2)))

        # print(original_cosine_similarity(array1, array2))
        # print(cosine_similarity(self.mapping.dimension_reduction(array1), self.mapping.dimension_reduction(array2)))

        return  after_oip, after_ip, ohd, hd, ojs, js, ocs, cs





def main():
    # before_oips = []
    # before_ips = []

    oips = []
    ips = []

    ohds = []
    hds = []

    ojss = []
    jss = []

    ocss = []
    css = []

    mse_ip = []
    mse_hd = []
    mse_js = []
    mse_cs = []

    number_of_features = []

    for i in range(10, 7000, 50):
        number_of_features.append(i)
        print("Iteration:", len(number_of_features))

        input_dimension = 5000
        insert_dimension = i
        arr1 = np.random.randint(0, 2, size=input_dimension)
        arr2 = np.random.randint(0, 2, size=input_dimension)

        features1 = np.random.randint(0, 2, size=insert_dimension)
        features2 = np.random.randint(0, 2, size=insert_dimension)

        # print(arr1, arr2)

        sparsity1 = 0
        sparsity2 = 0

        for i in range(arr1.size):
            if arr1[i] == 1:
                sparsity1 += 1
            if arr2[i] == 1:
                sparsity2 += 1

        sparsity = 1

        if sparsity1 > sparsity2:
            sparsity = sparsity1
        else:
            sparsity = sparsity2
        # sparsity = 1

        # print(inner_product(arr1, arr2))

        demo_operator = operator(input_dim=input_dimension, output_dim= sparsity**2, mapping_scheme=8)
        # print(demo_operator.mapping.output_dimension)
        a1, a2, b1, b2, c1, c2, d1, d2 = demo_operator.new_insert_feature(arr1,arr2, features1, features2, sparsity)

        oips.append(a1)
        ips.append(a2)

        ohds.append(b1)
        hds.append(b2)

        ojss.append(c1)
        jss.append(c2)

        ocss.append(d1)
        css.append(d2)

        # before_oips.append(a)
        # before_ips.append(b)

        # after_oips.append(c)
        # after_ips.append(d)

        # # print(c,d)

        # before_mse.append(abs(a-b))
        # after_mse.append(abs(c-d))

        # print(arr1, arr2)
        # demo_operator = operator(5, 2, 3)

        # arr1,arr2 = demo_operator.batch_insert_feature([1,3,4], arr1, arr2, [0,1,-1], [0,1,-1])
        # print ("After Insertions", arr1,arr2)


        # arr1,arr2 = demo_operator.batch_delete_feature([2,3,5], arr1, arr2)
        # print ("After Deletion", arr1,arr2)
    
    mse_ip = abs(np.array(oips) - np.array(ips))
    mse_hd = abs(np.array(ohds) - np.array(hds))
    mse_js = abs(np.array(ojss) - np.array(jss))
    mse_cs = abs(np.array(ocss) - np.array(css))

    fig, ax = plt.subplots(2, 2)
    
    ax[0][0].set_title('Inner Product')
    ax[0][1].set_title('Hamming Distance')
    ax[1][0].set_title('Jaccard Similarity')
    ax[1][1].set_title('Cosine Similarity')

    ax[0][0].set(xlabel='# features inserted', ylabel='Error in Similarity Measure')
    ax[0][1].set(xlabel='# features inserted', ylabel='Error in Similarity Measure')
    ax[1][0].set(xlabel='# features inserted', ylabel='Error in Similarity Measure')
    ax[1][1].set(xlabel='# features inserted', ylabel='Error in Similarity Measure')

    ax[0][0].plot(number_of_features, mse_ip, label="Error")
    ax[0][0].plot(number_of_features, oips, label="Original Measure")
    ax[0][0].plot(number_of_features, ips, label="Predicted Measure")

    ax[0][1].plot(number_of_features, mse_hd)
    ax[0][1].plot(number_of_features, ohds)
    ax[0][1].plot(number_of_features, hds)

    ax[1][0].plot(number_of_features, mse_js)
    ax[1][0].plot(number_of_features, ojss)
    ax[1][0].plot(number_of_features, jss)

    ax[1][1].plot(number_of_features, mse_cs)
    ax[1][1].plot(number_of_features, css)
    ax[1][1].plot(number_of_features, ocss)


    fig.legend()
    fig.tight_layout(pad=0.5)
    fig.set_figheight(9)
    fig.set_figwidth(9)

    
    #plt.show()
    fig.savefig('../Plots/insertion_all_plots1.png', orientation = 'landscape')

    # print(after_mse)
    # plt.xlabel("Number of Features inserted")
    # plt.ylabel("Error, Inner Product")
    # plt.plot(number_of_features, before_mse, label="MSE before insertion")
    # plt.plot(number_of_features, after_mse, label="MSE after insertion")

    # plt.plot(number_of_features, before_oips, label="Original IP before insertion")
    # plt.plot(number_of_features, before_ips, label="Predicted IP before insertion")

    # plt.plot(number_of_features, after_oips, label="Original IP after insertion")
    # plt.plot(number_of_features, after_ips, label="Predicted IP after insertion")
    # plt.legend()
    # plt.savefig("Insertion5.png")


if __name__ == "__main__":
    main()
