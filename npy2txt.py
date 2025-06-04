import numpy as np

policy_q = np.load('policy_q.npy')
flattened_array = policy_q.reshape(-1)
np.savetxt('policy_q.txt', flattened_array, fmt='%.6f')

print(policy_q)



