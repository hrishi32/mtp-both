from Object_Files.mapper import mapper
import numpy as np
from Object_Files.mapper2 import mapper as mapper2
from Object_Files.mapper3 import mapper as mapper3
from Object_Files.mapper4 import mapper as mapper4
from Object_Files.mapper5 import mapper as mapper5
from Object_Files.mapper6 import mapper as mapper6
from Object_Files.mapper7 import mapper as mapper7
from Object_Files.mapper8 import mapper as mapper8
from Object_Files.mapper9 import mapper as mapper9
"""
"""
# from mapper import mapper
# import numpy as np
# from mapper2 import mapper as mapper2
# from mapper3 import mapper as mapper3
# from mapper4 import mapper as mapper4
# from mapper5 import mapper as mapper5
# from mapper6 import mapper as mapper6
# from mapper7 import mapper as mapper7
# from mapper8 import mapper as mapper8
# from mapper9 import mapper as mapper9

# import random

def predict_ones(reduced_ones, size):
    numerator = np.log10(1-(reduced_ones/size))
    denominator = np.log10(1-(1/size))
    # print("size", size)
    # print("num-denom", numerator, denominator)
    return numerator/denominator

def count_ones(arr):
    result = 0

    for i in arr:
        if i == 1:
            result+=1
    
    return result
"""
    *
    * function inner_product(arr1,arr2)
    *
    * Summary: 
    *
    *   Given the 2 arrays, computes its inner product.
    *   
    * Parameters     : arr1 : Array
    *                  arr2 : Array
    *
    * Return Value  : Predicted value of Inner product.
    *
"""
def inner_product(arr1, arr2):
    reduced_ones1, reduced_ones2 = count_ones(arr1), count_ones(arr2)

    N = arr1.size
    # print("Reduced:", reduced_ones1, reduced_ones2)
    mod_a = predict_ones(reduced_ones1, N)
    mod_b = predict_ones(reduced_ones2, N)

    # print("mod_a:", mod_a, "mod_b:", mod_b)

    ip = 0.0

    for i in range(N):
        ip+=(arr1[i]*arr2[i])
    # print("N: ", N, "mod_a:", mod_a, "mod_b:", mod_b)
    v = ((1-(1.0/N))**mod_a) + ((1-(1.0/N))**mod_b) + (ip*1.0/N) -1.0
    if ((1-(1.0/N))**mod_a) + ((1-(1.0/N))**mod_b) + (ip*1.0/N) >=0.99 and ((1-(1.0/N))**mod_a) + ((1-(1.0/N))**mod_b) + (ip*1.0/N) <= 1.0:
        print('\n\n\n\n\n\n\n',((1-(1.0/N))**mod_a),'\t', ((1-(1.0/N))**mod_b),'\t', (ip*1.0/N),  '\n\n\n\n\n\n\n\n')
        v=abs(v)
        numerator = np.log10( v+0.001 )
    else:
         numerator = np.log10( v )

    denominator = np.log10(1-(1.0/N))

    denominator = abs(denominator)+0.001

    # if denominator == 0:
    #     denominator+=0.0000001

    # print("numerator", numerator, "denominator:", denominator)
    # print("num/denom", numerator/denominator)

    c = abs(numerator/denominator)

    return mod_a + mod_b - c

"""
    *
    * function hamming_distance(arr1,arr2)
    *
    * Summary: 
    *
    *   Given the 2 arrays, computes its hamming distance.
    *   
    * Parameters     : arr1 : Array
    *                  arr2 : Array
    *
    * Return Value  : Predicted value of hamming distance.
    *
"""

def hamming_distance(arr1, arr2):
    mod_a = count_ones(arr1)
    mod_b = count_ones(arr2)

    ip = inner_product(arr1, arr2)

    return mod_a + mod_b - (2*ip)
"""
    *
    * function jaccard_similarity(arr1,arr2)
    *
    * Summary: 
    *
    *   Given the 2 arrays, computes its Jaccard similarity.
    *   
    * Parameters     : arr1 : Array
    *                  arr2 : Array
    *
    * Return Value  : Predicted value of jaccard similarity.
    *
"""
def jaccard_similarity(arr1, arr2):
    ip = inner_product(arr1, arr2)
    hd = hamming_distance(arr1, arr2)

    return ip/(hd+ip)
"""
    *
    * function cosine_similarity(arr1,arr2)
    *
    * Summary: 
    *
    *   Given the 2 arrays, computes its cosine_similarity.
    *   
    * Parameters     : arr1 : Array
    *                  arr2 : Array
    *
    * Return Value  : Predicted value of cosine_similarity.
    *
"""
def cosine_similarity(arr1, arr2):
    mod_a = count_ones(arr1)
    mod_b = count_ones(arr2)
    ip = inner_product(arr1, arr2)

    return ip/((mod_a*mod_b)**0.5)
"""
    *
    * function inner_product(arr1,arr2)
    *
    * Summary: 
    *
    *   Given the 2 arrays, computes its inner product.
    *   
    * Parameters     : arr1 : Array
    *                  arr2 : Array
    *
    * Return Value  : Value of Inner product.
    *
"""
def original_inner_product(arr1, arr2):
    result = 0

    for i in range(arr1.size):
        result+=(arr1[i]*arr2[i])

    return result
"""
    *
    * function original_hamming_distance(arr1,arr2)
    *
    * Summary: 
    *
    *   Given the 2 arrays, computes its hamming distance.
    *   
    * Parameters     : arr1 : Array
    *                  arr2 : Array
    *
    * Return Value  : Value of Hamming distance.
    *
"""
def original_hamming_distance(arr1, arr2):
    result = 0

    for i in range(arr1.size):
        result+=(arr1[i]-arr2[i])

    return result
"""
    *
    * function original_jaccard_similarity(arr1,arr2)
    *
    * Summary: 
    *
    *   Given the 2 arrays, computes its jaccard similarity.
    *   
    * Parameters     : arr1 : Array
    *                  arr2 : Array
    *
    * Return Value  : Value of jaccard similarity.
    *
"""
def original_jaccard_similarity(arr1, arr2):
    numerator = inner_product(arr1, arr2)

    demoninator = 0
    for i, j in zip(arr1, arr2):
        demoninator += int(i|j)

    if demoninator == 0:
        return -1

    return numerator/demoninator
"""
    *
    * function original_cosine_similarity(arr1,arr2)
    *
    * Summary: 
    *
    *   Given the 2 arrays, computes its cosine_similarity.
    *   
    * Parameters     : arr1 : Array
    *                  arr2 : Array
    *
    * Return Value  : Value of cosine_similarity.
    *
"""
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


"""
    * class operator
    *
    * Summary of operator class:
    *
    *   This class provides functions to operate on two arrays via a associated mapping..
    *   Implemented methods support feature insertion, deletion and other functionalities.
    *
    * Description:
    *
    *   This class use a specific mapping to operate on arrays or data for different dimensionality reduction operations.
    *   Given the input array, it will be able to return a output array.
    *
"""
class operator:
    """
        * Summary of init function:
        *  
        *   It is only used while creating a new object. According to given parameters for mapping scheme, the
        *   particular mapping is assigned to its self variable "self.mapping"
        *
        * Parameters    : input_dim: integer
        *                 out_dim: integer
        *                 mapping_scheme: integer
        *
        * Description :
        *
        *   It creates a mapping array from input dimension to output dimension along with bits string.
        *   The output dimension is given as a parameter, however each mapping is responsible for choosing.
        *   its own output_dim.
        *   
    """
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
        elif mapping_scheme == 9:
            self.mapping = mapper9(input_dim=input_dim, out_dim=output_dim)  

    """
        *
        * function insert_feature(position=0, array1 = [], array2 = [], value1 = 0, value2 = 0)
        *
        * Summary: 
        *
        *   Changes a mapping for newly inserted feature in map array at given
        *   position. The insertion scheme depends on the mapping associated
        *   Inserts a value in the recquired position in the data arrays passed as the argument.
        *   
        *
        * Parameters     : position:Integer
        *                  array1: Array
        *                  array2: Array
        *                  value1: Real
        *                  value2: Real
        *
        * Return Value  : Data arrays 1 and 2 after feature insertion of value1 and value2 at the given position.
        *
    """
    def insert_feature(self, position=0, array1 = [], array2 = [], value1 = 0, value2 = 0):
        self.mapping.insert_feature(position=position)
        array1 = np.insert(array1, position, value1)
        array2 = np.insert(array2, position, value2)
        return array1,array2

    """
        *
        * function delete_feature(position=0, array1 = [], array2 = [])
        *
        * Summary: 
        *
        *   Deletes a mapping for deleted feature in map array at given
        *   position. The deletion scheme depends on the mapping associated.
        *   Deletes a value in the required position in the data arrays passed as the argument.
        *   
        * Parameters     : position:Integer
        *                  array1: Array
        *                  array2: Array
        *
        * Return Value  :  Data arrays 1 and 2 after feature deletion at the given position. 
        *
        * Description:
        *
        *   After execution of this function, input dimension will be reduced
        *   by 1, output dimension will remain same.
        *
    """
    def delete_feature(self, position=0, array1 = [], array2 = []):
        self.mapping.delete_feature(position=position)
        array1 = np.delete(array1, position)
        array2 = np.delete(array2, position)
        return array1, array2

    """
        *
        * function batch_insert_feature(batch_positions=[],array1=[],array2=[],batch_value1=[],batch_value2=[])
        *
        * Summary: 
        *
        *   Inserts a mapping for newly inserted features in map array at given
        *   position. Here, multiple features are inserted at once.
        *   Inserts the values as a batch in data arrays at the required batch positions.
        *
        * Parameters     : batch_positions: List of integers
        *                  array1 : Array
        *                  array2 : Array
        *                  batch_value1 : list of Real numbers
        *                  batch_value2 : list of Real numbers
        *
        * Return Value  : Data arrays 1 and 2 after feature insertion at given positions.
        *
        * Description:
        *
        *   When feature insertion in input vector is happened in batch, this method
        *   should be invoked. 
        *
    """

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
                array2 = np.insert(array2,i+factor,flags[i][1])
                factor+=1
                last_insertion +=1
                # flags = np.insert(flags, i, 0)
            elif len(flags[i]) != 0:
                array1 = np.insert(array1,i+factor,flags[i][0])
                array2 = np.insert(array2,i+factor,flags[i][1])
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


    """
        *
        * function batch_delete_feature(batch_positions=[],array1=[],array2=[])
        *
        * Summary: 
        *
        *   Deletes a mapping for deleted feature in map array at given
        *   positions. Here multiples features are deleted at once.
        *   Deletes the features as a batch in data arrays at the required batch positions.
        *   
        * Parameters     : batch_positions:List of integers
        *                  array1: Array
        *                  array2: Array
        *
        * Return Value  : Data arrays 1 and 2 after feature deletion.
        *
        * Description:
        *
        *   After execution of this function, input dimension will be reduced
        *   by number of batch positions, output dimension will depend on the mapping scheme.
        *
    """
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

    """
        *
        * function array_normalization(input_array)
        *
        * Summary: 
        *
        *   Given the input array, the function computes the normalized array.
        *   
        * Parameters     : input_array: List of real numbers
        *
        * Return Value  : output_array: List of real numbers
        *
        * Description:
        *
        *   This method is useful to get the normalized array.
        *
    """

    def array_normalization(self, input_array):
        array_norm = np.linalg.norm(input_array)
        # print ("array norm:",array_norm)
        result = np.zeros(input_array.size, dtype=float)
        for i in range(input_array.size):
            result[i] = (1.0*input_array[i])/array_norm

        return result

    """
        *
        * function inner_product(input_array1, input array2)
        *
        * Summary: 
        *
        *   Given the 2 arrays, computes its inner product.
        *   
        * Parameters     : input_array: List of real numbers
        *
        * Return Value  : output_array: List of real numbers
        *
    """

    def inner_product(self, input_array1, input_array2):
        # # input_array1 = self.array_normalization(input_array1)
        # # input_array2 = self.array_normalization(input_array2)

        # # print ("norm array1 :",input_array1)
        # # print ("norm array2 :",input_array2)

        output_array1 = self.mapping.dimension_reduction(input_array1)
        output_array2 = self.mapping.dimension_reduction(input_array2)

        result1 = original_inner_product(input_array1, input_array2)
        result2 = inner_product(output_array1, output_array2)

        # #print("Output1", output_array1)
        # #print("Output2", output_array2)

        # result1, result2 = 0, 0
        
        # for i, j in zip(input_array1, input_array2):
        #     result1+=(i*j)

        # for i, j in zip(output_array1, output_array2):
        #     result2+=(i*j)

        # #print("Input Inner Product:", result1)
        # #print("Output Inner Product:", result2)

        return result1, result2

    """
        *
        * function inner_product(input_array1, input array2)
        *
        * Summary: 
        *
        *   Given the 2 arrays, computes its inner product.
        *   
        * Parameters     : input_array: List of real numbers
        *
        * Return Value  : output_array: List of real numbers
        *
    """

    def jaccard_similarity(self, input_array1, input_array2):
        
        output_array1 = self.mapping.dimension_reduction(input_array1)
        output_array2 = self.mapping.dimension_reduction(input_array2)

        result1 = original_jaccard_similarity(input_array1, input_array2)
        result2 = jaccard_similarity(output_array1, output_array2)

        return result1, result2

    """
        *
        * function inner_product(input_array1, input array2)
        *
        * Summary: 
        *
        *   Given the 2 arrays, computes its inner product.
        *   
        * Parameters     : input_array: List of real numbers
        *
        * Return Value  : output_array: List of real numbers
        *
    """

    def cosine_similarity(self, input_array1, input_array2):
        
        output_array1 = self.mapping.dimension_reduction(input_array1)
        output_array2 = self.mapping.dimension_reduction(input_array2)

        result1 = original_cosine_similarity(input_array1, input_array2)
        result2 = cosine_similarity(output_array1, output_array2)

        return result1, result2

    # def Eucledian_distance(self, input_array1, input_array2):
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
    #         result1+=(i-j)**2
    #     result1 = sqrt(result1)

    #     for i, j in zip(output_array1, output_array2):
    #         result2+=(i-j)**2
    #     result2 = sqrt(result2)

        #print("Input Inner Product:", result1)
        #print("Output Inner Product:", result2)

        # return result1, result2

    # def Hammming_distance(self, input_array1, input_array2):
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
    #         result1+=abs(i-j)

    #     for i, j in zip(output_array1, output_array2):
    #         result2+=abs(i-j)

    #     #print("Input Inner Product:", result1)
    #     #print("Output Inner Product:", result2)

    #     return result1, result2

    """
        *
        * function get_feature_count()
        *
        * Summary: 
        *
        *   Calculates number of features mapped in each bin of output vector.
        *   
        * Parameters     : None
        *
        * Return Value  : feature_counter: List of integers
        *
        * Description:
        *
        *   Each number in the returned list indicates the number of features mapped at
        *   that position.
        *
    """
    def get_feature_counter(self):
        return self.mapping.get_feature_counter()

    """
        *
        * function get_feature_counter()
        *
        * Summary: 
        *
        *   Collects list of features mapped in each bin of output vector.
        *   
        * Parameters     : None
        *
        * Return Value  : feature_counter: List of list of integers
        *
        * Description:
        *
        *   Each list in the returned list indicates the positions of input vector
        *   that are mapped in output vector.
        *
    """
    def get_feature_count(self):
        return self.mapping.get_feature_count()

def main():
    arr1 = np.random.randint(0, 2, size=12419)
    arr2 = np.random.randint(0, 2, size=12419)
    print("Original Arrays", arr1, arr2)
    demo_operator = operator(12419, 2, 5)

    print(demo_operator.inner_product(arr1, arr2))
    # arr1,arr2 = demo_operator.batch_insert_feature([1,3,4], arr1, arr2, [0,1,-1], [0,2,-2])
    # print ("After Insertions", arr1,arr2)


    # arr1,arr2 = demo_operator.batch_delete_feature([2,3,5], arr1, arr2)
    # print ("After Deletion", arr1,arr2)

if __name__ == "__main__":
    main()