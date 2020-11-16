from clean_data import *
from sklearn import model_selection
from alg_desc_to_del import *
from model_log_reg_l2 import *
import matplotlib.pyplot as plt
import pickle
from random import sample
import random
import pandas as pd

"""
n_deletions: number of deletions
n_rounds: number of rounds to run to average out noise
m: l2 penalty of deletion
alpha: if given as input, the algo will be run until we are alpha-close to the optimizer of the regulrized loss.
iters: if alpha>0 is given, this is max number of iterations to check for alpha-closeness.
        Otherwise, this is "I" in our paper.
samprate: used to subsample full_adlut and mnist datasets.
perfect: flag for perfect unlearning. Use perfect=1 only with early stopping option.
eps_list: list of epsilon values to try. if perfect=1, use *only one* epsilon.
M: smoothness parameter of the loss function.
eta_list: list of learning rate values for line search. The first element is the one suggested by theory.
"""

random.seed(0)
np.random.seed(0)

n_deletions = 5
n_rounds = 10
m = 0.05
alpha = 0.001
iters = 25
samprate = 0.1
perfect  = 0
eps_list = [1]
#eps_list = [0.25, 0.4, 0.5, 0.75, 1, 10]
M = 0.5
eta_list = [1/(m+M)]
eta_list = eta_list + [0.01, 0.1, 0.5, 1, 5, 10]

""" Import Data Set """
X, y = clean_adult_full(scale_and_center=True, normalize=True, intercept=True, samprate = samprate)
#X, y = clean_adult(scale_and_center=True, normalize=True, intercept=True)
#X, y = clean_lawschool(scale_and_center=True, normalize=True, intercept=True)
#X, y = clean_mnist(d = 64, scale_and_center=True, intercept = False, normalize=True, samprate = samprate)

""" split train/test and create deletion sequence """
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
del_indices = np.random.randint(0, X_train.shape[0], size=n_deletions)
u_seq = [('-', ind,  X_train.iloc[ind], y_train.iloc[ind]) for ind in del_indices]

""" my attempt to include additions as well but I got some errors """
#X, y = clean_adult(scale_and_center=True, normalize=True, intercept=True)
#X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=.2)
#X_train = X_train.reset_index(drop=True)
#y_train = y_train.reset_index(drop=True)
#n_additions = 10
#n_deletions = 40
#add_indices = np.random.randint(0, X_train.shape[0], size=n_additions)
#add_seq = [('+', X_train.iloc[ind], pd.Series(y_train.iloc[ind])) for ind in add_indices]
#X_train = X_train.drop(index = add_indices)
#y_train = y_train.drop(index = add_indices)
#del_indices = np.random.randint(0, X_train.shape[0], size=n_deletions)
#del_seq = [('-', ind,  X_train.iloc[ind], y_train.iloc[ind]) for ind in del_indices]
#u_seq = sample(add_seq+del_seq, len(add_seq+del_seq,))

unlearning_acc = {}
for eps in eps_list:
    unlearning_acc[eps] = []
retrain_acc = []
update_iter_seq = []
training_iter_seq = []

for i in range(n_rounds):
    print('round', i+1, 'of', n_rounds)
    """ Deletion """
    desc_del_algorithm = DescDel(X_train, X_test, y_train, y_test, epsilon=eps_list, delta=1.0/np.power(len(y_train), 1),
                         model_class=LogisticReg, update_sequence=u_seq,
                        l2_penalty=m, eta = eta_list, iters = iters, alpha = alpha, retrain_flag = 0, perfect = perfect)
    desc_del_algorithm.run()
    t_seq = desc_del_algorithm.iter_seq
    t_max = max(t_seq[1:len(t_seq)])
    update_iter_seq.append(t_max)
    training_iter_seq.append(t_seq[0])
    for eps in eps_list:
        unlearning_acc[eps].append( np.array(desc_del_algorithm.model_accuracies[eps]) )
    """ Retrain Baseline """
    desc_del_algorithm = DescDel(X_train, X_test, y_train, y_test, epsilon= [1], delta = 1.0/np.power(len(y_train), 1),
                        model_class=LogisticReg, update_sequence=u_seq,
                        l2_penalty=0, eta = eta_list, iters = t_max, retrain_flag = 1)
    desc_del_algorithm.run()
    retrain_acc.append( np.array(desc_del_algorithm.model_accuracies[1]) )

unlearning = {}
for eps in eps_list:
    unlearning[eps] = sum(unlearning_acc[eps])/float(n_rounds)
retrain = sum(retrain_acc)/float(n_rounds)
avg_update_iter_seq = sum(update_iter_seq)/len(update_iter_seq)
avg_training_iter_seq = sum(training_iter_seq)/len(training_iter_seq)

print('average number of iterations for training:', avg_training_iter_seq)
print('average number of iterations for deletion:', avg_update_iter_seq)

for eps in eps_list:
    plt.plot(unlearning[eps], label=r'$\epsilon={}$'.format(eps))
plt.plot(retrain, color='black', label='retrain baseline')
plt.xlabel('update number')
plt.ylabel('test accuracy')
plt.legend(loc = 'best', prop={'size': 10})
plt.title(r'lawschool dataset, $\alpha = {}$'.format(alpha))

""" SAVE RESULTS IF YOU LIKE: Update the name of the data set used """
#pickle.dump([unlearning, retrain, avg_training_iter_seq, avg_update_iter_seq],
#                open('pickles/{}_lawschool_alpha_{}_m_{}_iters_{}_perfect_{}_ndeletion_{}_nrounds_{}.p'.format(
#                        samprate, alpha, m, iters, perfect, n_deletions, n_rounds), 'wb'))
#plt.savefig('figures/BLAH.png', dpi = 300)

plt.savefig('plot.png', dpi=300)
plt.show()
