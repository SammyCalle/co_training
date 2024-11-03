import numpy as np
import pandas as pd
import random
import copy


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

    def nst_cotraining(self, X1, X2, y, key):

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
        self.clf1_.partial_fit(X1.iloc[L], y.iloc[L])
        self.clf2_.partial_fit(X2.iloc[L], y.iloc[L])

        # loop until we have assigned labels to everything in U or we hit our iteration break condition
        while it != self.k_ and U:
            it += 1

            y1_prob = self.clf1_.predict_proba(X1[U_])
            y2_prob = self.clf2_.predict_proba(X2[U_])

            n, p = [], []

            for i in (y1_prob[:, 0].argsort())[-self.n_:]:
                if y1_prob[i, 0] > 0.5:  # 0.5 would be a threshold so no lower than 0.5 will be accepted
                    n.append(i)
            for i in (y1_prob[:, 1].argsort())[-self.p_:]:
                if y1_prob[i, 1] > 0.5:
                    p.append(i)

            for i in (y2_prob[:, 0].argsort())[-self.n_:]:
                if y2_prob[i, 0] > 0.5:
                    n.append(i)
            for i in (y2_prob[:, 1].argsort())[-self.p_:]:
                if y2_prob[i, 1] > 0.5:
                    p.append(i)

            # Label the samples and remove the newly added samples from U_
            y[[U_[x] for x in p]] = 1
            y[[U_[x] for x in n]] = 0

            New_Labeled  = U_[p + n]

            U_ = [elem for elem in U_ if elem not in p + n]  # Remove elements from U_ that are in p or n
            num_to_add = len(p) + len(n)
            # Add new elements to U_
            U_ += U[-num_to_add:]  # Directly append the last num_to_add elements from U

            #Fit the new labeled data
            self.clf1_.partial_fit(X1[New_Labeled], y[New_Labeled], classes=[0, 1])
            self.clf2_.partial_fit(X2[New_Labeled], y[New_Labeled], classes=[0, 1])

        self.model_X1_dict[key] = self.clf1_
        self.model_X2_dict[key] = self.clf2_
