import numpy as np
import pdb
import sys

class DescDel:
    """Implement Algorithm 1 from Descent-to-Delete"""
    def __init__(self, X_train, X_test, y_train, y_test, epsilon, delta, model_class,
                 update_sequence, l2_penalty=0, eta = None, iters = 1, alpha = -1, retrain_flag = 0, perfect = 0):
        """
        sigma: noise added to guarantee (eps, delta) deletion
        loss_fn_constants: smoothness, strong convexity, lipschitz constant
        loss_fn_gradient: fn that given X, y, theta returns grad(f(X,y, theta))
        update_sequence: list of tuples [(x_1, y_1, +), (x_2, y_2, -), etc]
        update_grad_iter: number of allowed gradient iterations per round
        """
        self.X_train = X_train
        self.X_u = X_train
        self.y_train = y_train
        self.y_u = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.epsilon = epsilon
        self.delta = delta
        self.sigma = 0
        self.gamma = 0
        self.model_class = model_class
        self.update_sequence = update_sequence
        self.start_grad_iter = iters
        self.update_grad_iter = iters
        self.max_iters = iters
        self.datadim = X_train.shape[1]
        self.n = len(self.y_train)
        self.l2_penalty = l2_penalty
        self.eta = eta
        self.retrain_flag = retrain_flag
        self.perfect = perfect
        self.alpha = alpha
        self.models = []
        self.iter_seq = []
        self.noisy_models = {}
        self.model_accuracies = {}
        for eps in self.epsilon:
            self.noisy_models[eps] = []
            self.model_accuracies[eps] = []

    def update(self, update):
        """Given update, output retrained model, noisy and secret state"""
        self.update_data_set(update)
        if self.retrain_flag:
            new_model = self.train(iters = self.max_iters, init=None)
            self.models.append(new_model)
            self.model_accuracies[self.epsilon[0]].append(self.get_test_accuracy(new_model))
        else:
            if self.perfect == 0:
                new_model = self.train(iters = self.max_iters, init=self.models[-1])
                self.models.append(new_model)
                for eps in self.epsilon:
                    self.set_sigma(eps)
                    noisy_model = self.publish(new_model)
                    self.noisy_models[eps].append(noisy_model)
                    self.model_accuracies[eps].append(self.get_test_accuracy(noisy_model))
            else:
                for eps in self.epsilon:
                    new_model = self.train(iters = self.max_iters, init=self.noisy_models[eps][-1])
                    self.set_sigma(eps)
                    noisy_model = self.publish(new_model)
                    self.noisy_models[eps].append(noisy_model)
                    self.model_accuracies[eps].append(self.get_test_accuracy(noisy_model))

    def train(self, iters, init=None):
        if init:
            model = self.model_class(init.theta, l2_penalty=self.l2_penalty, eta = self.eta)
        else:
            par = np.random.normal(0,1, self.datadim)
            par = par/(np.sqrt(np.sum(np.power(par, 2))))
            par = np.random.uniform(0,1)*par
            model = self.model_class(par, l2_penalty=self.l2_penalty, eta = self.eta)
        if self.retrain_flag:
            for _ in range(iters):
                model.proj_gradient_step(self.X_u, self.y_u, grad = None, grad_flag = 0)
        else:
            for t in range(iters+1):
                grad = model.gradient_loss_fn(self.X_u, self.y_u)
                if self.get_distance_to_opt(grad) <= self.alpha:
                    self.iter_seq.append(t)
                    break
                if t==iters:
                    sys.exit("max iteration is hit, re-run!")
                model.proj_gradient_step(self.X_u, self.y_u, grad, grad_flag = 1)
                if (t==iters - 1 and self.alpha == -1):
                    self.iter_seq.append(t+1)
                    break
        return model

    def publish(self, model):
        noise = np.random.normal(0, self.sigma, self.datadim)
        theta = model.theta + noise
        if np.sum(np.power(theta, 2)) > 1:
            norm = np.sqrt(np.sum(np.power(theta, 2)))
            theta = theta/norm
        return self.model_class(theta, l2_penalty=self.l2_penalty, eta=self.eta)

    def update_data_set(self, update):
        """Update X_u, y_u with update (+, x, y) or (-, index, x, y)."""
        self.X_u = self.X_u.reset_index(drop=True)
        self.y_u = self.y_u.reset_index(drop=True)
        if update[0] == '-':
            try:
                self.X_u = self.X_u.drop(update[1])
                self.y_u = self.y_u.drop(update[1])
            except:
                pdb.set_trace()
        if update[0] == '+':
            self.X_u = self.X_u.append(update[1])
            self.y_u = self.y_u.append(update[2])

    def run(self):
        if self.retrain_flag == 0:
            self.set_gamma()
            self.set_Delta()
            if self.alpha == -1:
                self.set_start_grad_iter()
        initial_model = self.train(iters = self.start_grad_iter, init=None)
        self.models.append(initial_model)
        if self.retrain_flag == 0:
            for eps in self.epsilon:
                self.set_sigma(eps)
                initial_noisy_model = self.publish(initial_model)
                self.noisy_models[eps].append(initial_noisy_model)
                self.model_accuracies[eps].append(self.get_test_accuracy(initial_noisy_model))
        else:
            self.model_accuracies[self.epsilon[0]].append(self.get_test_accuracy(initial_model))
        for update in self.update_sequence:
            self.update(update)

    def get_test_accuracy(self, model):
        y_hat = model.predict(self.X_test)
        return np.float(np.sum([np.array(y_hat) == np.array(self.y_test)]))/np.float(len(self.y_test))

    def get_distance_to_opt(self, grad):
        grad_norm = np.sqrt(np.sum(np.power(grad, 2)))
        alpha = (2/self.l2_penalty)*grad_norm
        return alpha

    def set_gamma(self):
        if self.eta:
            self.gamma = 1 - self.eta[0]*self.l2_penalty
        else:
            dummy_model = self.model_class(theta = 0, l2_penalty=self.l2_penalty, eta = self.eta)
            loss_fn_constants = dummy_model.get_constants()
            self.gamma = (loss_fn_constants['smooth']-loss_fn_constants['strong'])/(loss_fn_constants['strong'] +
                                                                                     loss_fn_constants['smooth'])

    def set_Delta(self):
        if self.alpha == -1:
            dummy_model = self.model_class(theta = 0, l2_penalty=self.l2_penalty, eta = self.eta)
            loss_fn_constants = dummy_model.get_constants()
            Delta_num = (8*loss_fn_constants['lip']*np.power(self.gamma, self.update_grad_iter))
            Delta_denom = (loss_fn_constants['strong']*self.n*(1-np.power(self.gamma, self.update_grad_iter)))
            self.Delta = Delta_num/Delta_denom
        else:
            self.Delta = 2*self.alpha

    def set_sigma(self, eps):
        sigma_num = self.Delta/np.sqrt(2)
        sigma_denom = np.sqrt(np.log(1/self.delta) + eps) - np.sqrt(np.log(1/self.delta))
        self.sigma = sigma_num/sigma_denom

    def set_start_grad_iter(self):
        dummy_model = self.model_class(theta = 0, l2_penalty=self.l2_penalty, eta = self.eta)
        loss_fn_constants = dummy_model.get_constants()
        T = self.update_grad_iter + (np.log(loss_fn_constants['diameter']*loss_fn_constants['strong']*len(self.y_train)/(2*loss_fn_constants['lip']))/np.log(1/self.gamma))
        self.start_grad_iter = int(np.ceil(T))
