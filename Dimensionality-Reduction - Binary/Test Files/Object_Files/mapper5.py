import numpy as np
import random
# Without Compensation Mapping
class mapper:
    def __init__(self, input_dim = 50, out_dim = 15):
        print("You are in Mapper 5!")
        self.input_dimension = input_dim
        self.output_dimension = self.find_compression_length()

        print ("Generating Mapping. Please wait")

        self.map = np.zeros(input_dim,dtype=int)
        # self.feature_counter = []
        # for i in range(out_dim):
        #     self.feature_counter.append([])

        for i in range(input_dim):
            alpha = random.randint(0,self.output_dimension-1)
            self.map[i] = alpha
            # self.feature_counter[alpha].append(i)
        print ("Mapping generated")
        # self.get_mapping_info()
        # print ("Mapping :",self.map)
        # print ("Feature :",self.feature_counter)

        #print("Initializing...\n", "Bits:", self.bits, "\nMap:", self.map)

    def find_compression_length(self):
        return int(0.1*self.input_dimension)+10

    def insert_feature(self, position=0):
        # print ("Inserting new feature at the ",position,"of data.")
        if position <= self.input_dimension:
            self.input_dimension += 1
            alpha = random.randint(0,self.output_dimension-1)
            self.map = np.insert(self.map, position,alpha,axis=0)
            # print (self.map)
            # updated_feature_counter_array = []
            # for i in range(self.input_dimension):
            #     if self.map[i][alpha] == 1:
            #         updated_feature_counter_array.append(i)
            # self.feature_counter[alpha] = updated_feature_counter_array
        else :
            print("Feature position is incorrect !")
        # print("Inserting New Feature at position:", position)
        # print("Bits:", self.bits)
        # print("Map:", self.map)
        # print("feature_counter :",self.feature_counter)

    def delete_feature(self, position=0):
        # print ("Pos:",position)
        if position < self.input_dimension:
            # beta=0
            # for i in range(len(self.map[position])):
            #     if self.map[position][i] == 1:
            #         beta = i
            #         break
            # print ("beta:",beta)
            self.input_dimension -= 1
            self.map = np.delete(self.map, position,axis=0)

            # updated_feature_counter_array = []
            # for i in range(self.input_dimension):
            #     if self.map[i][beta] == 1:
            #         updated_feature_counter_array.append(i)
            # self.feature_counter[beta] = updated_feature_counter_array

            # print (self.feature_counter[beta])


        else :
            print("Feature position is incorrect !")
        # print("Deleted Index:", position)
        # print("Maping Changed for position:", alpha)
        # print("Bits:", self.bits)
        # print("Map:", self.map)
        # print("feature_counter :",self.feature_counter)

    def batch_insert_feature(self,batch_positions=[]):
        flags = np.zeros(self.input_dimension)
        for i in range(len(batch_positions)):
            flags[batch_positions[i]] = 1
            # self.insert_feature(position=batch_positions[i])

        i = 0
        factor = 0
        old_dim = self.input_dimension
        last_insertion = 0
        # print ("start")
        while i < old_dim:

            # print (i,flags[i])
            if flags[i] == 1 and last_insertion == 0 :
                self.insert_feature(i+factor)
                factor+=1
                last_insertion +=1
                # flags = np.insert(flags, i, 0)
                # i += 1
            elif flags[i] == 1:
                self.insert_feature(i+factor)
                factor+=1
                last_insertion+=1
            else:
                last_insertion = 0
            
            i+=1
        # print ("end")

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
                # flags = np.insert(flags, i, 0)
                # i += 1
            elif flags[i] == 1:
                self.delete_feature(i-factor)
                factor+=1
                last_deletion+=1
            else:
                last_deletion = 0
            
            i+=1

        # print (self.get_mapping_info())


        # for i in range(len(batch_positions)):
        #     self.delete_feature(position=batch_positions[i])

    def dimension_reduction(self, input_array):
        # output_array = np.zeros(self.output_dimension, dtype=float)

        # for i in range(self.input_dimension):
        #     if self.map[i] != -1:
        #         output_array[self.map[i]] += (self.bits[i])*input_array[i]
        output_array = np.zeros(self.output_dimension, dtype=float)

        for i in range(self.input_dimension):
            output_array[self.map[i]] += input_array[i]

        for i in range(self.output_dimension):
            if output_array[i]>0:
                output_array[i]=1

        return output_array


        # return output_array

    def input_dim(self):
        return self.input_dim

    def output_dim(self):
        return self.output_dim

    def get_feature_count(self):
        arr = self.get_feature_counter()
        feature_counter = np.zeros(self.output_dimension)

        for i in range(len(arr)):
            feature_counter[i] += len(arr[i])

        return feature_counter

        # return self.feature_counter

    def get_feature_counter(self):
        feature_count = []
        temp = []
        for i in range(self.output_dimension):
            feature_count.append(temp)
            temp = []
        
        # print("Input D")
        # print(self.map)
        for i in range(self.input_dimension):
            feature_count[self.map[i]].append(i)

        # print (feature_count)
        return feature_count

    def get_mapping_info(self):
        print ("Input Features:",self.input_dimension)
        print ("Output Features:",self.output_dimension)
        print ("Features Distribution:",self.get_feature_counter())
        print ("Features Distribution Count:",self.get_feature_count())
        print("Map", self.map)





def main():

    demomap = mapper(input_dim=100,out_dim=3)
    demomap.get_mapping_info()
    arr1 = np.random.randint(0, 99, 11)
    arr2 = np.random.randint(0,2,100)
    print("Original Array:", arr2)
    print(demomap.dimension_reduction(arr2))
    demomap.batch_delete_feature(arr1)
    demomap.get_mapping_info()


if __name__ == "__main__":
    main()