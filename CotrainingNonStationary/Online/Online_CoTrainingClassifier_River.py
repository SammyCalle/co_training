import random
import copy
from river import forest, stream
import numpy as np


class Non_Stationary_CoTrainingClassifier(object):
    """
	Parameters:
	clf - The classifier that will be used in the cotraining algorithm on the X1 feature set
		(Note a copy of clf will be used on the X2 feature set if clf2 is not specified).

	clf2 - (Optional) A different classifier type can be specified to be used on the X2 feature set
		 if desired.

	p - (Optional) The number of positive examples that will be 'labeled' by each classifier during each iteration
		The default is the is determined by the smallest integer ratio of positive to negative samples in L (from paper)

	n - (Optional) The number of negative examples that will be 'labeled' by each classifier during each iteration
		The default is the is determined by the smallest integer ratio of positive to negative samples in L (from paper)

	k - (Optional) The number of iterations
		The default is 30 (from paper)

	u - (Optional) The size of the pool of unlabeled samples from which the classifier can choose
		Default - 75 (from paper)
	"""

    def __init__(self, clf, clf2=None, p=-1, n=-1, k=30, u=75):
        self.clf1_ = clf

        # we will just use a copy of clf (the same kind of classifier) if clf2 is not specified
        if clf2 == None:
            self.clf2_ = copy.copy(clf)
        else:
            self.clf2_ = clf2

        # if they only specify one of n or p, through an exception
        if (p == -1 and n != -1) or (p != -1 and n == -1):
            raise ValueError('Current implementation supports either both p and n being specified, or neither')

        self.p_ = p
        self.n_ = n
        self.k_ = k
        self.u_ = u
        self.model_X1_dict = {}
        self.model_X2_dict = {}

        random.seed()

    def online_cotraining(self, X1, X2, y, key):

        """
        Description:
        fits the classifiers on the partially labeled data, y.

        Parameters:
        X1 - array-like (n_samples, n_features_1): first set of features for samples
        X2 - array-like (n_samples, n_features_2): second set of features for samples
        y - array-like (n_samples): labels for samples, -1 indicates unlabeled
        """
        # we need y to be a numpy array so we can do more complex slicing
        # y = np.asarray(y)

        # set the n and p parameters if we need to
        if self.p_ == -1 and self.n_ == -1:
            # num_pos = (y['Malware'] == 1).sum()
            # num_neg = (y['Malware'] == 0).sum()
            num_pos = sum(1 for y_i in y if y_i == 1)
            num_neg = sum(1 for y_i in y if y_i == 0)

            n_p_ratio = num_neg / float(num_pos)

            if n_p_ratio > 1:
                self.p_ = 1
                self.n_ = round(self.p_ * n_p_ratio)

            else:
                self.n_ = 1
                self.p_ = round(self.n_ / n_p_ratio)

        assert (self.p_ > 0 and self.n_ > 0 and self.k_ > 0 and self.u_ > 0)

        # the set of unlabeled samples

        U = [i for i, y_i in enumerate(y) if y_i == -1]

        # we randomize here, and then just take from the back so we don't have to sample every time
        random.shuffle(U)

        # this is U' in paper
        U_ = U[-min(len(U), self.u_):]

        # the samples that are initially labeled
        L = [i for i, y_i in enumerate(y) if y_i != -1]

        # remove the samples in U_ from U
        U = U[:-len(U_)]

        it = 0  # number of cotraining iterations we've done so far


        # The initial fit for the labeled data
        initial_stream_X1 = stream.iter_pandas(X1.iloc[L], y[L])
        for x, label in initial_stream_X1:
            self.clf1_.learn_one(x, label)

        initial_stream_X2 = stream.iter_pandas(X2.iloc[L], y[L])
        for x, label in initial_stream_X2:
            self.clf2_.learn_one(x, label)

        # loop until we have assigned labels to everything in U or we hit our iteration break condition
        while it != self.k_ and U:
            it += 1
            y1_prob = []
            y2_prob = []

            proba_stream_X1 = stream.iter_pandas(X1.iloc[U_], y[U_])
            for x, label in proba_stream_X1:
                y1_prob.append(self.clf1_.predict_proba_one(x))

            proba_stream_X2 = stream.iter_pandas(X2.iloc[U_], y[U_])
            for x, label in proba_stream_X2:
                y2_prob.append(self.clf2_.predict_proba_one(x))

            values_0_x1 = [d[0] for d in y1_prob]
            values_1_x1 = [d[1] for d in y1_prob]

            array_x1 = np.array([values_0_x1, values_1_x1]).T

            values_0_x2 = [d[0] for d in y2_prob]
            values_1_x2 = [d[1] for d in y2_prob]

            array_x2 = np.array([values_0_x2, values_1_x2]).T

            n, p = [], []

            for i in (array_x1[:, 0].argsort())[-self.n_:]:
                if array_x1[i, 0] > 0.5:  # 0.5 would be a threshold so no lower than 0.5 will be accepted
                    n.append(i)
            for i in (array_x1[:, 1].argsort())[-self.p_:]:
                if array_x1[i, 1] > 0.5:
                    p.append(i)

            for i in (array_x2[:, 0].argsort())[-self.n_:]:
                if array_x2[i, 0] > 0.5:
                    n.append(i)
            for i in (array_x2[:, 1].argsort())[-self.p_:]:
                if array_x2[i, 1] > 0.5:
                    p.append(i)

            # Label the samples and remove the newly added samples from U_
            y[[U_[x] for x in p]] = 1
            y[[U_[x] for x in n]] = 0

            labeled_index = p+n
            new_labeled = []
            for index in labeled_index:
                new_labeled.append(U_[index])

            U_ = [elem for elem in U_ if elem not in p + n]  # Remove elements from U_ that are in p or n

            # Add new elements to U_
            add_counter = 0
            num_to_add = len(p) + len(n)
            while add_counter != num_to_add and U:
                add_counter += 1
                U_.append(U.pop())

            #Fit the new labeled data
            end_stream_X1 = stream.iter_pandas(X1.iloc[new_labeled], y[new_labeled])
            for x, label in end_stream_X1:
                self.clf1_.learn_one(x, label)

            end_stream_X2 = stream.iter_pandas(X2.iloc[new_labeled], y[new_labeled])
            for x, label in end_stream_X2:
                self.clf1_.learn_one(x, label)



        self.model_X1_dict[key] = self.clf1_
        self.model_X2_dict[key] = self.clf2_
