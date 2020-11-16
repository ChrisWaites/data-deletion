import numpy as np
import pandas as pd
import clean_data
import pdb
import copy

class LogisticReg:
    """Implement Algorithm 1 from Descent-to-Delete"""

    def __init__(self, theta, l2_penalty=0, eta = None):
        self.l2_penalty = l2_penalty
        self.eta = eta
        self.theta = theta
        self.constants_dict = {'strong': self.l2_penalty, 'smooth': 0.5 + self.l2_penalty, 'diameter': 2.0,
                               'lip': 1.0 + 2*self.l2_penalty}

    def loss_fn(self, theta, X, y):
        n = X.shape[0]
        log_loss = (1/n)*np.sum(np.log(1 + np.exp(-y*np.dot(X, theta)))) + 0.5*self.l2_penalty*np.sum(np.power(theta, 2))
        return log_loss

    def gradient_loss_fn(self, X, y):
        n = X.shape[0]
        log_grad = np.dot(np.diag(-y/(1 + np.exp(y*np.dot(X, self.theta)))), X)
        log_grad_sum = np.dot(np.ones(n), log_grad)
        reg_grad = self.l2_penalty*self.theta
        return (reg_grad + (1/n)*log_grad_sum)

    def get_constants(self):
        return self.constants_dict

    def proj_gradient_step(self, X, y, grad = None, grad_flag = 0):
        #if self.eta:
        #    eta = self.eta
        #else:
        #    eta = 2.0/(self.constants_dict['strong'] + self.constants_dict['smooth'])
        start = 1
        old_loss = 0
        for eta in self.eta: #line search
            current_theta = self.theta
            if grad_flag == 0:
                grad = self.gradient_loss_fn(X, y)
            next_theta = current_theta - eta*grad
            if np.sum(np.power(next_theta, 2)) > 1:
                next_theta = next_theta/(clean_data.l2_norm(next_theta))
            new_loss = self.loss_fn(next_theta, X, y)
            if (new_loss < old_loss or start == 1):
                start = 0
                best_eta = eta
                best_theta = next_theta
                old_loss = new_loss
        self.theta = best_theta

    def predict(self, X):
        probs = 1.0/(1+np.exp(-np.dot(X, self.theta)))
        return pd.Series([1 if p >= .5 else -1 for p in probs])

