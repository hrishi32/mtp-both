import numpy as np
import random
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
        *   The output dimension is given as a parameter.
        *   
    """
    def __init__(self, input_dim = 50, out_dim = 15):
        self.input_dimension = input_dim
        self.output_dimension = out_dim
        self.bits = np.random.randint(-1, high= 1, size= input_dim)

        for i in range(self.bits.size):
            if self.bits[i] == 0:
                self.bits[i] = 1

        self.map = np.random.randint(0, high= out_dim, size = input_dim)
        #print("Initializing...\n", "Bits:", self.bits, "\nMap:", self.map)

    """
        *
        * function insert_feature(position=0)
        *
        * Summary: 
        *
        *   Inserts a mapping for newly inserted feature in map array at given
        *   position. 
        *   Note: Feature insertion scheme used is "no compensation".
        *   
        *
        * Parameters     : position:Integer
        *
        * Return Value  : Nothing -- Note: It changes map array internally.
        *
    """

    def insert_feature(self, position=0):
        if position < self.input_dimension:
            self.input_dimension += 1
            self.bits = np.insert(self.bits, position, (random.randint(0,1)-0.5)*2)
            self.map = np.insert(self.map, position, random.randint(0,self.output_dimension-1))
        else :
            print("Feature position is incorrect !")
        #print("Inserting New Feature at position:", position)
        #print("Bits:", self.bits)
        #print("Map:", self.map)

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
            beta = self.map[position]
            self.bits = np.delete(self.bits, position)
            self.map = np.delete(self.map, position)
            alpha = random.randint(0, self.input_dimension-1)
            while self.map[alpha] == beta:
                alpha = random.randint(0, self.input_dimension-1)
            self.bits[alpha] = (random.randint(0,1)-0.5)*2
            self.map[alpha] = beta
        else :
            print("Feature position is incorrect !")
        #print("Deleted Index:", position)
        #print("Maping Changed for position:", alpha)
        #print("Bits:", self.bits)
        #print("Map:", self.map)

    """
        *
        * function batch_insert_feature(batch_positions=[])
        *
        * Summary: 
        *
        *   Inserts a mapping for newly inserted features in map array at given
        *   position. Here, features are inserted in batch.
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
        for i in range(len(batch_positions)):
            self.insert_feature(position=batch_positions[i])

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
        for i in range(len(batch_positions)):
            self.delete_feature(position=batch_positions[i])

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
            output_array[self.map[i]] += (self.bits[i])*input_array[i]

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




def main():
    # #print (np.random.randint(0,high=1,size=50))
    demomap = mapper()
    demomap.insert_feature()
    demomap.delete_feature()

    temp = np.random.randint(-256, 256, demomap.input_dimension)
    #print(temp)

    #print(demomap.dimension_reduction(temp))

if __name__ == "__main__":
    main()