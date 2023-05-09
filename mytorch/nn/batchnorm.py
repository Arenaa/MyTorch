import numpy as np


class BatchNorm1d:

    def __init__(self, num_features, alpha=0.9):

        self.alpha = alpha
        self.eps = 1e-8

        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        # Running mean and variance, updated during training, used during
        # inference
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the
        training phase of the problem or the inference phase.
        So see what values you need to recompute when eval is False.
        """
        self.Z = Z
        self.N = Z.shape[0]
        self.M = (1/self.N) * np.sum(Z, axis=0)
        self.V = (1/self.N) * np.sum((self.Z - self.M)**2, axis=0)

        if eval == False:
            # training mode
            self.NZ = (self.Z - self.M) / np.sqrt(self.V **2 + self.eps)
            self.BZ = (self.NZ * self.BW) + self.Bb

            self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.M
            self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.V
        else:
            # inference mode
            self.NZ = (self.Z - self.running_M) / np.sqrt( self.running_V + self.eps)
            self.BZ = (self.NZ * self.BW) + self.Bb

        return self.BZ

    def backward(self, dLdBZ):

        self.dLdBW = np.sum(dLdBZ)
        self.dLdBb = None  # TODO

        dLdNZ = None  # TODO
        dLdV = None  # TODO
        dLdM = None  # TODO

        dLdZ = None  # TODO

        return NotImplemented
