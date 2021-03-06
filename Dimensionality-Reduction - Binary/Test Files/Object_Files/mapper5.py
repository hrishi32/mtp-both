import numpy as np
import random
# Without Compensation Mapping

"""
    * class mapper
    *
    * Summary of mapper class:
    *
    *   This class stores the mapping between input vector and output vector.
    *   Implemented methods support feature insertion, deletion and other functionalities.
    *
    * Description:
    *
    *   This class creates a mapping between input vector and output vector when initiated.
    *   Given the input array, it will be able to return a output array.
    *
"""

class mapper:
    """
        * Summary of init function:
        *  
        *   It is only used while creating a new object. According to given parameters, the random
        *   mapping from input dimension 'd' to output dimension 'k' is created.
        *
        * Parameters    : input_dim: integer
        *                 out_dim: integer
        *
        * Description :
        *
        *   It creates a mapping array from input dimension to output dimension along with bits string.
        *   The output dimension is given as a parameter, however we compute it using method 'find_compression_length'.
        *   
    """

    def __init__(self, input_dim = 50, out_dim = 15):
        print("You are in Mapper 5!")
        self.input_dimension = input_dim
        self.output_dimension = self.find_compression_length()

        self.map = np.zeros(input_dim,dtype=int)
        
        for i in range(input_dim):
            alpha = random.randint(0,self.output_dimension-1)
            self.map[i] = alpha
            
        print ("Mapping generated")
        
    """
        *
        * function find_compression_length()
        *
        * Summary:
        *   
        *   Calculates appropriate output dimension (compression length) from input dimension.
        *
        * Parameters    : None -- Note: It uses input dimension from global variable.
        *
        * Return Value  : Output Dimension
        *
    """

    def find_compression_length(self):
        return int(0.001*self.input_dimension)+1500

    """
        *
        * function insert_feature(position=0)
        *
        * Summary: 
        *
        *   Inserts a mapping for newly inserted feature in map array at given
        *   position. 
        *   Note: As this mapper is only for deletion, it does not implement bin
        *   expansion here.
        *
        * Parameters     : position:Integer
        *
        * Return Value  : Nothing -- Note: It changes map array internally.
        *
    """

    def insert_feature(self, position=0):
        if position <= self.input_dimension:
            self.input_dimension += 1
            alpha = random.randint(0,self.output_dimension-1)
            self.map = np.insert(self.map, position,alpha,axis=0)
            
        else :
            print("Feature position is incorrect !")
        
    """
        *
        * function delete_feature(position=0)
        *
        * Summary: 
        *
        *   Deletes a mapping for deleted feature in map array at given
        *   position. The deletion scheme is 'No Compensation'
        *   
        * Parameters     : position:Integer
        *
        * Return Value  : Nothing -- Note: It changes map array internally.
        *
        * Description:
        *
        *   After execution of this function, input dimension will be reduced
        *   by 1, output dimension will remain same.
        *
    """

    def delete_feature(self, position=0):
        if position < self.input_dimension:
        
            self.input_dimension -= 1
            self.map = np.delete(self.map, position,axis=0)

        else :
            print("Feature position is incorrect !")
        
    """
        *
        * function batch_insert_feature(batch_positions=[])
        *
        * Summary: 
        *
        *   Inserts a mapping for newly inserted features in map array at given
        *   position. Here, features are inserted in batch.
        *   Note: As this mapper is only for deletion, it does not implement bin
        *   expansion here.
        *
        * Parameters     : batch_positions: List of integers
        *
        * Return Value  : Nothing -- Note: It changes map array internally.
        *
        * Description:
        *
        *   When feature insertion in input vector is happened in batch, this method
        *   should be invoked. 
        *
    """

    def batch_insert_feature(self,batch_positions=[]):
        flags = np.zeros(self.input_dimension)
        for i in range(len(batch_positions)):
            flags[batch_positions[i]] = 1
        
        i = 0
        factor = 0
        old_dim = self.input_dimension
        last_insertion = 0
    
        while i < old_dim:

            # print (i,flags[i])
            if flags[i] == 1 and last_insertion == 0 :
                self.insert_feature(i+factor)
                factor+=1
                last_insertion +=1
                
            elif flags[i] == 1:
                self.insert_feature(i+factor)
                factor+=1
                last_insertion+=1
            else:
                last_insertion = 0
            
            i+=1
        

    """
        *
        * function batch_delete_feature(batch_positions=[])
        *
        * Summary: 
        *
        *   Deletes a mapping for deleted feature in map array at given
        *   positions. The deletion scheme is 'No Compensation'
        *   
        * Parameters     : batch_positions:List of integers
        *
        * Return Value  : Nothing -- Note: It changes map array internally.
        *
        * Description:
        *
        *   After execution of this function, input dimension will be reduced
        *   by number of batch positions, output dimension will remain same.
        *
    """

    def batch_delete_feature(self,batch_positions=[]):
        flags = np.zeros(self.input_dimension)
        for i in range(len(batch_positions)):
            flags[batch_positions[i]] = 1

        i = 0
        factor = 0
        old_dim = self.input_dimension
        last_deletion = 0
        # print ("start")
        while i < old_dim:

            # print (i,flags[i])
            if flags[i] == 1 and last_deletion == 0 :
                self.delete_feature(i-factor)
                factor+=1
                last_deletion +=1
                
            elif flags[i] == 1:
                self.delete_feature(i-factor)
                factor+=1
                last_deletion+=1
            else:
                last_deletion = 0
            
            i+=1

    """
        *
        * function dimension_reduction(input_array)
        *
        * Summary: 
        *
        *   Given the input array, the function computes the associated output array
        *   
        * Parameters     : input_array: List of real numbers
        *
        * Return Value  : output_array: List of real numbers
        *
        * Description:
        *
        *   This method is useful to get the output array form associated mapping.
        *
    """

    def dimension_reduction(self, input_array):
        output_array = np.zeros(self.output_dimension, dtype=float)

        for i in range(self.input_dimension):
            output_array[self.map[i]] += input_array[i]

        for i in range(self.output_dimension):
            if output_array[i]>0:
                output_array[i]=1

        return output_array

    """
        *
        * function input_dim()
        *
        * Summary: 
        *
        *   Method to get input dimension
        *   
        * Parameters     : None
        *
        * Return Value  : self.input_dim: Integer
        *
    """

    def input_dim(self):
        return self.input_dim

    """
        *
        * function output_dim()
        *
        * Summary: 
        *
        *   Method to get output dimension
        *   
        * Parameters     : None
        *
        * Return Value  : self.output_dim: Integer
        *
    """

    def output_dim(self):
        return self.output_dim

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

    def get_feature_count(self):
        arr = self.get_feature_counter()
        feature_counter = np.zeros(self.output_dimension)

        for i in range(len(arr)):
            feature_counter[i] += len(arr[i])

        return feature_counter


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

    def get_feature_counter(self):
        feature_count = []
        temp = []
        for i in range(self.output_dimension):
            feature_count.append(temp)
            temp = []
        
        for i in range(self.input_dimension-1):
            feature_count[self.map[i]].append(i)

        # print (feature_count)
        return feature_count

    """
        *
        * function get_mapping_info()
        *
        * Summary: 
        *
        *   A function to print mapping information.
        *   
        * Parameters     : None
        *
        * Return Value  : Nothing -- Note: Prints associated mapping information on console.
        *
        * Description:
        *
        *   This method is useful for debugging purposes.
        *
    """

    def get_mapping_info(self):
        print ("Input Features:",self.input_dimension)
        print ("Output Features:",self.output_dimension)
        print ("Features Distribution:",self.get_feature_counter())
        print ("Features Distribution Count:",self.get_feature_count())
        print("Map", self.map)





def main():
    # #print (np.random.randint(0,high=1,size=50))
    demomap = mapper(input_dim=7,out_dim=3)
    demomap.get_mapping_info()
    arr1 = np.random.randint(0, 10, 7)
    print("Original Array:", arr1)
    print(demomap.dimension_reduction(arr1))


    # demomap.get_mapping_info()
    # # demomap.insert_feature()
    # # demomap.insert_feature()
    # # demomap.insert_feature()
    # # demomap.insert_feature()
    # demomap.delete_feature(position=3)
    # demomap.get_mapping_info()
    # demomap.delete_feature(position=4)
    # # demomap.insert_feature()
    demomap.batch_delete_feature([1,3,4])
    # print (demomap.get_feature_count())
    demomap.get_mapping_info()
    # temp = np.random.randint(-256, 256, demomap.input_dimension)
    #print(temp)

    #print(demomap.dimension_reduction(temp))

if __name__ == "__main__":
    main()