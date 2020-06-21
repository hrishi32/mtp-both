import numpy as np
import random
# With Compensation -> 1 step
class mapper:
    def __init__(self, input_dim = 50, out_dim = 15):
        print ("You are using mapper 6.")
        self.input_dimension = input_dim
        self.output_dimension = self.find_compression_length()


        self.map = np.zeros(input_dim,dtype=int)
        # self.feature_counter = []
        # for i in range(out_dim):
        #     self.feature_counter.append([])

        self.map = np.random.randint(0, high=self.output_dimension, size=self.input_dimension, dtype=int)

        # for i in range(input_dim):
        #     alpha = random.randint(0,self.output_dimension-1)
        #     self.map[i] = alpha
            # self.feature_counter[alpha].append(i)
        print ("Mapping generated")
        # print ("Mapping :",self.map)
        # print ("Feature :",self.feature_counter)

        #print("Initializing...\n", "Bits:", self.bits, "\nMap:", self.map)
       

    def find_compression_length(self):
        return int(0.1*self.input_dimension)+10

    def insert_feature(self, position=0):
        self.input_dimension+=1

        if self.output_dimension == self.find_compression_length():
            
            # print ("Inserting new feature at the ",position,"of data.")
            if position <= self.input_dimension:
                # self.input_dimension += 1
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

        else:

            # temp_map = np.random.randint(0, high=self.output_dimension+1, size = self.output_dimension, dtype=int)
            # temp_map = np.array(range(self.output_dimension))
            # # print("tempmap",temp_map)
            # for i in range(self.input_dimension-1):
            #     # print(self.map[i])
            #     self.map[i] = temp_map[self.map[i]]
            # # print("map",self.map)
            # self.output_dimension = self.find_compression_length()
            # # print ("outdim",self.output_dimension)
            if position <= self.input_dimension:
                # self.input_dimension += 1
                # self.get_mapping_info()
                alpha = random.randint(0,self.output_dimension-1)
                self.map = np.insert(self.map, position,alpha,axis=0)

                feature_selected = int(self.input_dimension/self.output_dimension)
                for i in range(feature_selected):
                    alpha = random.randint(0,self.input_dimension-1)
                    # print("alpha_val:",alpha)
                    self.map[alpha]=self.output_dimension
                self.output_dimension+=1
            

                # self.get_mapping_info()
                # print (self.map)
                # updated_feature_counter_array = []
                # for i in range(self.input_dimension):
                #     if self.map[i][alpha] == 1:
                #         updated_feature_counter_array.append(i)
                # self.feature_counter[alpha] = updated_feature_counter_array
            else :
                print("Feature position is incorrect !")
            # print("Inserting New Feature at position:", position)

        # print("outdim", self.output_dimension)
        # print("map:")
        # for i in self.map:
        #     if i >= self.output_dimension:
        #         print(i, ", "),



    def delete_feature(self, position=0):
        # print ("position to be deleted:",position)
        if position < self.input_dimension:
            beta=self.map[position]
            # print ("Copressed feature is non-uniform:",beta)
            # print ("beta:",beta)
            alpha = random.randint(0,self.input_dimension-1)
            count = 0
            while self.map[alpha] == beta:
                alpha = random.randint(0,self.input_dimension-1)
                if count > 10 :
                    break
                count += 1
            # print ("mapping from :",alpha,"is compensated to:",beta)
            self.map[alpha] = beta
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
        # print("Update feature_counter :",self.get_feature_counter())

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
                self.insert_feature(i+factor)#-last_insertion)
                factor+=1
                last_insertion+=1
            else:
                last_insertion = 0
            
            i+=1
            #self.get_mapping_info()
        # print ("end")

    def batch_delete_feature(self,batch_positions=[]):
        flags = np.zeros(self.input_dimension)
        for i in range(len(batch_positions)):
            flags[batch_positions[i]] = 1

        i = 0
        factor = 0
        old_dim = self.input_dimension
        last_deletion = 0
        # print ("starting deletion")
        while i < old_dim:

            # print ("-",i,flags[i])
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
        for i in range(self.input_dimension-1):
            # print("i", i)
            # print("map_i", self.map)
            # print("lenfc", len(feature_count))
            # print("out_dim", self.output_dimension)
            # print("funoutdim", self.find_compression_length())
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
    # #print (np.random.randint(0,high=1,size=50))
    demomap = mapper(input_dim=100,out_dim=3)
    demomap.get_mapping_info()
    arr1 = np.random.randint(0, 10, 100)
    print("Original Array:", arr1)
    print("Reduced array:",demomap.dimension_reduction(arr1))
    batch_pos=[]
    for i in range(11):
        batch_pos.append(random.randint(0,99))
    demomap.batch_insert_feature(batch_pos)


    # demomap.get_mapping_info()
    # # demomap.insert_feature()
    # # demomap.insert_feature()
    # # demomap.insert_feature()
    # # demomap.insert_feature()
    
    # # demomap.insert_feature()
    # demomap.batch_delete_feature([1,3,4])
    # print (demomap.get_feature_count())
    demomap.get_mapping_info()
    # temp = np.random.randint(-256, 256, demomap.input_dimension)
    #print(temp)

    #print(demomap.dimension_reduction(temp))

if __name__ == "__main__":
    main()