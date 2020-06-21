from Object_Files.mapper import mapper, np
from Object_Files.basic_operator import operator
# from ... import *

def main():
    input_size, output_size = 50000, 2000
    demo_operator = operator(input_dim=input_size, output_dim=output_size)

    arr1 = np.random.randint(0, high=50, size=input_size)
    arr2 = np.random.randint(0, high=50, size=input_size)

    #print("Array1:", arr1)
    #print("Array2:", arr2)

    demo_operator.inner_product(arr1, arr2)

    arr1 = np.insert(arr1, 1, 2)
    arr2 = np.insert(arr2, 1, 3)

    demo_operator.insert_feature(1, arr1, arr2)

    demo_operator.inner_product(arr1, arr2)

    arr1 = np.delete(arr1, 0)
    arr2 = np.delete(arr2, 0)

    demo_operator.delete_feature(0,arr1, arr2)

    demo_operator.inner_product(arr1, arr2)

if __name__ == "__main__":
    main()
