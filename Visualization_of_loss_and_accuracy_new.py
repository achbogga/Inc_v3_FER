
# coding: utf-8

# In[1]:

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats


# In[2]:

def remove_first_line(log_file):
    with open(log_file, 'r') as fin:
        data = fin.read().splitlines(True)
    with open(log_file, 'w') as fout:
        fout.writelines(data[1:])

def plot_l_and_a(log_file):
    t = np.loadtxt(log_file, delimiter=',')
    training_accuracy, = plt.plot(t[:,1],'r', label='training_accuracy')
    validation_accuracy, = plt.plot(t[:,3],'b', label='validation_accuracy')
    plt.legend(handles=[training_accuracy, validation_accuracy], bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
    plt.show()
    training_loss, = plt.plot(t[:,2],'r', label='training_loss')
    validation_loss, = plt.plot(t[:,4],'b', label='validation_loss')
    plt.legend(handles=[training_loss, validation_loss], bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
    plt.show()


# In[3]:

plot_l_and_a('basic_cnn_training_200_log')


# In[13]:

remove_first_line('basic_cnn_training_16000_log')


# In[5]:

plot_l_and_a('basic_cnn_training_250_log')


# In[6]:

plot_l_and_a('basic_cnn_training_1000_log')


# In[7]:

plot_l_and_a('basic_cnn_training_20000_log')


# In[8]:

t = np.loadtxt('basic_cnn_training_20000_log', delimiter=',')
print np.unravel_index(t[:,3].argmax(), t[:,3].shape)


# In[9]:

from copy import deepcopy
a = list(deepcopy(t[:,4]))
a, size = sorted(a), len(a)
res = [a[i + 1] - a[i] for i in xrange(size) if i+1 < size]
print "MinDiff: {0}, MaxDiff: {1}.".format(min(res), max(res))


# In[10]:

print np.unravel_index(t[:,4].argmin(), t[:,4].shape)


# In[11]:

print "final validation accuracy is: "+ str(100*t[:,3][7303])+ "%"


# In[12]:

plot_l_and_a('basic_cnn_training_7303_log')


# In[14]:

plot_l_and_a('basic_cnn_training_16000_log')


# In[ ]:



