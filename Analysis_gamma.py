#Analysis gamma

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

data = np.loadtxt('gamma_1000.txt')
data = data.reshape(-1)-1
print(data.shape)

plt.figure(figsize=(12,4))
# matplotlib histogram
plt.hist(data, density=True,color = 'blue', edgecolor = 'black',
         bins = int(180/2))

locs, _ = plt.yticks() 
print(locs)
plt.yticks(locs,np.round(locs/len(data),3))

#plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
# Add labels
plt.title('Probability Histogram of Gamma')
plt.xlabel('Gamma')
plt.ylabel('Probability')

plt.show()