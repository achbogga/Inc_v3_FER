
# coding: utf-8

# In[1]:

import numpy as np
from matplotlib import pyplot as plt


# In[2]:

training_log = np.loadtxt('training_log_Inc_v3_10000.txt', delimiter=',')


# In[41]:

training_accuracy, = plt.plot(training_log[:,1][:100],'r', label='training_accuracy')
validation_accuracy, = plt.plot(training_log[:,3][:100],'b', label='validation_accuracy')
plt.legend(handles=[training_accuracy, validation_accuracy])
plt.show()


# In[4]:

print training_log[:,2]


# In[26]:

print training_log[:,3][1917]


# In[25]:

print np.unravel_index(training_log[:,3].argmax(), training_log[:,3].shape)


# In[32]:

plt.plot(training_log[:,2][:2000],'r',training_log[:,4][:2000],'b')


# In[28]:

plt.show()


# In[34]:

plt.plot(training_log[:,2][:100],'r',training_log[:,4][:100],'b')
plt.show()


# In[ ]:



