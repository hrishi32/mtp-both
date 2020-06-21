from Object_Files.mapper import mapper, np
from Object_Files.basic_operator import operator
import matplotlib.pyplot as plt

def sample_error(N, M):
    sample_size=100
    arr1 = np.random.randint(0, high=N*2, size= N)
    arr2 = np.random.randint(0, high=N*2, size= N)
    average_sample_error = 0
    for i in range(sample_size):
        demo_operator = operator(N, M)

        inner_product1, inner_product2 = demo_operator.inner_product(arr1, arr2)

        average_sample_error+=((inner_product1-inner_product2)**2)

    
    average_sample_error/=sample_size
    average_sample_error = average_sample_error**0.5
    
    return average_sample_error


def main():
    # epochs = 500

    N = 100

    while N <= 500:
        print("N: ", N)
        error = np.zeros(N, dtype=float)
        for M in range(1, N):
            print("--epoch #", M)
            error[M-1] = sample_error(N, M)

        plt.plot(range(N), error)
        N+=50

    plt.show()

if __name__ == "__main__":
    main()

