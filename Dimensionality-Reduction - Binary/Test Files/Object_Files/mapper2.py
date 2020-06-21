import numpy as np
import random

class mapper:
    def __init__(self, input_dim = 50, out_dim = 15):
        self.input_dimension = input_dim
        self.output_dimension = out_dim
        self.bits = np.random.randint(-1, high= 1, size= input_dim)

        for i in range(self.bits.size):
            if self.bits[i] == 0:
                self.bits[i] = 1

        self.map = np.random.randint(0, high= out_dim, size = input_dim)
        #print("Initializing...\n", "Bits:", self.bits, "\nMap:", self.map)

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

    def delete_feature(self, position=0):
        if position < self.input_dimension:
            self.input_dimension -= 1
            self.bits = np.delete(self.bits, position)
            self.map = np.delete(self.map, position)
        else :
            print("Feature position is incorrect !")
        #print("Deleted Index:", position)
        #print("Maping Changed for position:", alpha)
        #print("Bits:", self.bits)
        #print("Map:", self.map)

    def batch_insert_feature(self,batch_positions=[]):
        for i in range(len(batch_positions)):
            self.insert_feature(position=batch_positions[i])

    def batch_delete_feature(self,batch_positions=[]):
        for i in range(len(batch_positions)):
            self.delete_feature(position=batch_positions[i])

    def dimension_reduction(self, input_array):
        output_array = np.zeros(self.output_dimension, dtype=float)

        for i in range(self.input_dimension):
            output_array[self.map[i]] += (self.bits[i])*input_array[i]

        return output_array

    def input_dim(self):
        return self.input_dim

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