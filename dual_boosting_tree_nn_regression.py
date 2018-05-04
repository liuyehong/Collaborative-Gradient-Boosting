import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from data_generator_regression import *
from scipy.optimize import fmin


class dual_boosting_regression():

    def __init__(self, n_estimator=10, max_depth=1, hidden_layer_sizes=(200,), learning_rate=0.1, labda=1):
        self.n_estimator = n_estimator
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.hidden_layer_sizes = hidden_layer_sizes
        self.labda = labda


    def fit(self, X, y):
        self.list_trees = []
        self.list_nns = []
        boosted_predict_tree = np.zeros(len(y))
        boosted_predict_nn = np.zeros(len(y))
        for k in range(self.n_estimator):
            #print k

            # compare two methods
            if k == 0:
                I = np.random.binomial(1, 0.5, size=1)[0]

            else:
                if np.sum((y-boosted_predict_tree)**2) - np.sum((y-boosted_predict_nn)**2) > 0:
                    I = 1
                    print 'nn is better'
                else:
                    I = 0
                    print 'tree is better'

            #tree step
            if k == 0:
                boosted_predict_tree = np.zeros(len(y))
            else:
                boosted_predict_tree = self.learning_rate*np.sum(np.array([t.predict(X) for t in self.list_trees]), axis=0)


            boosted_res_tree = y - boosted_predict_tree + I*self.labda*(boosted_predict_nn - boosted_predict_tree)

            # build new tree
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, boosted_res_tree)
            self.list_trees.append(tree)



            #nn step

            if k == 0:
                boosted_predict_nn = np.zeros(len(y))
            else:
                boosted_predict_nn = self.learning_rate*np.sum(np.array([t.predict(X) for t in self.list_nns]), axis=0)

            boosted_res_nn = y - boosted_predict_nn + (1-I)*self.labda*(boosted_predict_tree - boosted_predict_nn)

            # build new nn
            nn = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes)
            nn.fit(X, boosted_res_nn)
            self.list_nns.append(nn)




    def predict_tree(self, X):
        list_trees_predict = self.learning_rate*np.sum(np.array([tree.predict(X) for tree in self.list_trees]), axis=0)
        return list_trees_predict

    def predict_nn(self, X):
        list_nn_predict = self.learning_rate*np.sum(np.array([nn.predict(X) for nn in self.list_nns]), axis=0)
        return list_nn_predict

    def predict_bag(self, X):
        list_trees_predict = self.learning_rate * np.sum(np.array([tree.predict(X) for tree in self.list_trees]),
                                                         axis=0)
        list_nn_predict = self.learning_rate * np.sum(np.array([nn.predict(X) for nn in self.list_nns]), axis=0)
        return 0.5*(list_trees_predict + list_nn_predict)


    def mse_tree(self, X, y):
        return np.var(y-self.predict_tree(X))


    def mse_nn(self, X, y):
        return np.var(y-self.predict_nn(X))

    def mse_bag(self, X, y):
        return np.var(y-self.predict_bag(X))





if __name__ == '__main__':
    X, y = f(n=2000, p=2, std=0.1)
    X_train, y_train = X[:1000, :], y[:1000]
    X_test, y_test = X[1000:, :], y[1000:]
    boost = dual_boosting_regression(n_estimator=100, labda=1, learning_rate=0.1)
    boost.fit(X_train, y_train)

    print boost.mse_tree(X_test, y_test)
    print boost.mse_nn(X_test, y_test)
    print boost.mse_bag(X_test, y_test)







