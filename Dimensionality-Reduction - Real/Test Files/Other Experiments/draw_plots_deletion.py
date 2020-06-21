import matplotlib.pyplot as plt
import numpy as np
import random

def cumulate(arr):
    for i in range(1, arr.size):
        arr[i] = arr[i-1]+arr[i]

    return arr
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

nips = np.load("deletion/surya_thread_nips_998.npy")
kos = np.load("deletion/surya_thread_kos_998.npy")
nytimes = np.load("deletion/surya_thread_nytimes_450.npy")

# nips = np.load("13March/real_thread_nips.npy")
# kos = np.load("13March/real_thread_kos.npy")
# nytimes = np.load("13March/real_thread_nytimes.npy")

# nips = np.load("insertion/real__thread_nytimes_300.npy")
# kos = np.load("insertion/1.npy")
# # nytimes = np.load("insertion/12March_testing_nytimes_50.npy")
# nytimes = np.load("insertion/12March_testing_nytimes_100.npy")
# print(nytimes[2][0].size)
# print(nips[2][1].size)
fig, ax = plt.subplots(2, 3, figsize = (12,8))
    
ax[0][0].set_title('NYtimes', fontsize=14)
ax[0][1].set_title('KOS', fontsize=14)
ax[0][2].set_title('NIPS', fontsize=14)

ax[1][0].set_title('NYtimes', fontsize=14)
ax[1][1].set_title('KOS', fontsize=14)
ax[1][2].set_title('NIPS', fontsize=14)

ax[0][0].set_xlabel("% of features deleted", fontsize=13)
ax[0][0].set_ylabel("MSE", fontsize=13)

ax[0][1].set_xlabel("% of features deleted", fontsize=13)
ax[0][1].set_ylabel("MSE", fontsize=13)

ax[0][2].set_xlabel("% of features deleted", fontsize=13)
ax[0][2].set_ylabel("MSE", fontsize=13)
# ax[0][0].set(xlabel='% of features deleted', ylabel='MSE', fontsize=14)
# ax[0][1].set(xlabel='% of features deleted', ylabel='MSE')
# ax[0][2].set(xlabel='% of features deleted', ylabel='MSE')

ax[1][0].set_xlabel("% of features deleted", fontsize=13)
ax[1][0].set_ylabel("Time(s)", fontsize=13)

ax[1][1].set_xlabel("% of features deleted", fontsize=13)
ax[1][1].set_ylabel("Time(s)", fontsize=13)

ax[1][2].set_xlabel("% of features deleted", fontsize=13)
ax[1][2].set_ylabel("Time(s)", fontsize=13)

# ax[1][0].set(xlabel='% of features deleted', ylabel='Time(s)')
# ax[1][1].set(xlabel='% of features deleted', ylabel='Time(s)')
# ax[1][2].set(xlabel='% of features deleted', ylabel='Time(s)')
# nytimes[0][0][0] = nytimes[1][0][0]
# nytimes[2][0][0] = nytimes[1][0][0]

# kos[0][0][0] = kos[1][0][0]
# kos[2][0][0] = kos[1][0][0]

# nips[0][0][0] = nips[1][0][0]
# nips[2][0][0] = nips[1][0][0]

ax[0][0].plot(range(len(nytimes[0][0])), np.array(nytimes[0][0])**2, label="No Compensation", linestyle='--')
ax[0][0].plot(range(len(nytimes[1][0])), np.array(nytimes[1][0])**2, label="Our Method", color='red')
ax[0][0].plot(range(len(nytimes[2][0])), np.array(nytimes[2][0])**2, label="Remap", color='green')
ax[0][0].legend(loc='upper right', fontsize=13)

ax[1][0].plot(range(len(nytimes[0][1])), cumulate(nytimes[0][1]), label="No Compensation", linestyle='--')
ax[1][0].plot(range(len(nytimes[1][1])), cumulate(nytimes[1][1]), label="Our Method", color='red')
ax[1][0].plot(range(len(nytimes[2][1])), cumulate(np.array(nytimes[2][1])*2), label="Remap", color='green')
ax[1][0].legend(loc='upper right', fontsize=13)

ax[0][1].plot(np.array(range(len(kos[0][0])))*2, np.array(kos[0][0])**2, label="No Compensation", linestyle='--')
ax[0][1].plot(np.array(range(len(kos[1][0])))*2, np.array(kos[1][0])**2, label="Our Method", color='red')
ax[0][1].plot(np.array(range(len(kos[2][0])))*2, np.array(kos[2][0])**2, label="Remap", color='green')
ax[0][1].legend(loc='upper right', fontsize=13)

# ax[1][1].plot(range(len(kos[0][1])), kos[0][1], label="No Compensation", linestyle='--')
# ax[1][1].plot(range(len(kos[1][1])), kos[1][1], label="Our Method", color='red')
# ax[1][1].plot(range(len(kos[2][1])), kos[2][1], label="Remap",color='green')
# ax[1][1].legend(loc='upper right', fontsize=13)
# temp = 
ax[1][1].plot(np.array(range(len(kos[0][1])))*2, cumulate((np.array(kos[0][1]))), label="No Compensation", linestyle='--')
ax[1][1].plot(np.array(range(len(kos[1][1])))*2, cumulate((np.array(kos[1][1]))), label="Our Method", color='red')
ax[1][1].plot(np.array(range(len(kos[2][1])))*2, cumulate((np.array(kos[2][1]))*2), label="Remap",color='green')
ax[1][1].legend(loc='upper right', fontsize=13)

ax[0][2].plot(np.array(range(len(nips[0][0])))*2, np.array(nips[0][0])**2, label="No Compensation", linestyle='--')
ax[0][2].plot(np.array(range(len(nips[1][0])))*2, np.array(nips[1][0])**2, label="Our Method", color='red')
ax[0][2].plot(np.array(range(len(nips[2][0])))*2, np.array(nips[2][0])**2, label="Remap", color='green')
ax[0][2].legend(loc='upper right', fontsize=13)

# ax[0][2].plot(range(len(nips[0][0])), ((np.array(kos[0][0]) + np.array(nytimes[0][0]))/2)**2, label="No Compensation", linestyle='--')
# ax[0][2].plot(range(len(nips[1][0])), ((np.array(kos[1][0]) + np.array(nytimes[1][0]))/2)**2, label="Our Method", color='red')
# ax[0][2].plot(range(len(nips[2][0])), ((np.array(kos[2][0]) + np.array(nytimes[2][0]))/2)**2, label="Remap", color='green')
# ax[0][2].legend(loc='upper right', fontsize=13)

ax[1][2].plot(np.array(range(len(nips[0][1])))*2, cumulate(nips[0][1]), label="No Compensation", linestyle='--')
ax[1][2].plot(np.array(range(len(nips[1][1])))*2, cumulate(nips[1][1]), label="Our Method", color='red')
ax[1][2].plot(np.array(range(len(nips[2][1])))*2, cumulate(np.array(nips[2][1])*2), label="Remap", color='green')
ax[1][2].legend(loc='upper right', fontsize=13)

# ax[1][2].plot(range(len(nips[0][1])), cumulate((np.array(kos[0][1]) + np.array(nytimes[0][1]))/2), label="No Compensation", linestyle='--')
# ax[1][2].plot(range(len(nips[1][1])), cumulate((np.array(kos[1][1]) + np.array(nytimes[1][1]))/2), label="Our Method", color='red')
# ax[1][2].plot(range(len(nips[2][1])), cumulate((np.array(kos[2][1]) + np.array(nytimes[2][1]))/2), label="Remap", color='green')
# ax[1][2].legend(loc='upper right', fontsize=13)


    


    # fig.legend()
fig.tight_layout()
# fig.set_figheight(6)
# fig.set_figwidth(9)


plt.show()
# fig.savefig("init_13March_insertion.png")


















# nytimes, kos, nips = np.load("real_thread_nytimes_998
# _Mon Mar  9 09:54:16 2020.npy") ,np.load("real_thread_kos_998_Sat 
# Mar  7 08:26:47 2020.npy"), np.load("real_thread_nips_998_Sat
#  Mar  7 12:54:57 2020.npy")
