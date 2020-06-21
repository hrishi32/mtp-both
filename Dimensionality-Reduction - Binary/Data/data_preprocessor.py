from os.path import abspath, exists
import numpy as np

data_array = []
def load_data():
	f_path = abspath("docword.enron.txt")
	if exists(f_path):
		with open(f_path) as f:
			datapoints = int(f.readline())
			features = int(f.readline())
			unique_words = int(f.readline())
			last_num = 1
			feature_array = np.zeros(features,dtype=int)
			count = 0
			while True:
				line = f.readline() 
				if line:
					words = line.split()
					num = int(words[0])
					position = int(words[1])
					count = int(words[2])
					if num == last_num:
						feature_array[position-1] = count
					else:
						data_array.append(feature_array)
						count += 1
						feature_array = np.zeros(features,dtype=int)
						last_num = num
						feature_array[position-1] = count

					if count > 100 :
						break



						
				else:
					break


def main():
	load_data()
	print (data_array[0])

if __name__ == '__main__':
	main()
