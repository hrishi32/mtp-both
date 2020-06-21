from mapper import mapper, np
from basic_operator import operator
# import random

class batch_operator:
    def __init__(self, input_dim=50, output_dim = 15):
        self.operator = operator(input_dim=input_dim, output_dim=output_dim)
       


    def insert_feature(self, batch_position=[], batch_array1 = [], batch_array2 = [], batch_value1 = [], batch_value2 = []):
        for i in range(batch_position.size):
            self.operator.insert_feature(position=batch_position[i], array1 = batch_array1[i], array2 = batch_array2[i], value1 = batch_value1[i], value2 = batch_value2[i])



    def delete_feature(self, batch_position=[], batch_array1 = [], batch_array2 = []):
        for i in range(batch_position.size):
            self.operator.delete_feature(position=batch_position[i], array1 = batch_array1[i], array2 = batch_array2[i])

    def batch_inner_product(self, batch_array1=[], batch_array2=[]):
        results = []
        for i in range(len(batch_array1)):
            result_1, result_2 = self.operator.inner_product(batch_array1[i],batch_array2[i])
            results.append(result_1,result_2)
        return results

def main():
    operator = batch_operator()

if __name__ == '__main__':
    main()

