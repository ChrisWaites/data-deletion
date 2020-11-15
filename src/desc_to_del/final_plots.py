from clean_data import *
from sklearn import model_selection
from alg_desc_to_del import *
from model_log_reg_l2 import *
import matplotlib.pyplot as plt
import pickle
from random import sample
import pandas as pd

n_deletions = 50
L = 100
m = 0.04
alpha = 0.008
samprate = 1
#eps_list = [0.25, 0.3, 0.4, 0.5, 1]
eps_list = [0.3, 0.4, 0.5, 1]

adult_results = pickle.load( open('new_pickles/{}_lawschool_alpha_{}_m_{}_ndeletion_{}_L_{}.p'.format(samprate, alpha, m, n_deletions, L), "rb") )
unlearning = adult_results[0]
retrain = adult_results[1]
avg_training_iter_seq = adult_results[2]
avg_update_iter_seq = adult_results[3]

for eps in eps_list:
    plt.plot(unlearning[eps], label=r'$\epsilon={}$'.format(eps))
plt.plot(retrain, '--', color='black', label='retrain baseline')
plt.xlabel('update number')
plt.ylabel('test accuracy')
plt.ylim(0.69, 0.81)
plt.title(r'lawschool dataset, $m = {}$, $\alpha = {}$'.format(m, alpha))
plt.legend(loc = 'best', prop={'size': 10})
#plt.savefig('figures/lawschool_var_iter.png', dpi = 300)
print('average number of iterations for training:', avg_training_iter_seq)
print('average number of iterations for deletion:', avg_update_iter_seq)
plt.show()
