# thanks to Daniel Kidd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
import scipy

# Transform step can take time. Looking into alternative techniques that avoid np.dot

class MCA(BaseEstimator, TransformerMixin):

    def __init__(self, K, var_cutoff=98, correction='benzecri', n_samples=100000):
        self.K = K                    # Number of features (before dummy variable encoding)
        self.var_cutoff = var_cutoff  # Percentage of explained variance to keep in truncated basis
        self.n_samples = n_samples    # number of samples to use for analysis
        if correction is not None and correction not in ['benzecri', 'greenacre']:
            raise ValueError('correction type must be None, benzecri, or greenacre')
        else:
            self.correction = correction

        self.m = None  # Number of rows in data
        self.J = None  # Number of features after dummy variable encoding
        self.k = 0     # Number of reduced factors to include in new basis
        self.E = None  # Eigenvalues of reduced basis
        self.S = None  # Eigenvalues of Burt matrix
        self.F = None  # Principal coordinates, rows
        self.G = None  # Principal coordinates, columns
        self.expl_var = None  # Explained variance of reduced basis eigenvectors
        self.tot_expl_var = np.nan  # Total explained variance in truncated basis
        self.keep_columns = None  # list of boolean values indicating which columns to keep
        # self.categories = None  # List of category labels considered in MCA
        # self.removed_cols = None  # List of category labels not considered in MCA (no examples in provided sample)

    def fit(self, data, y=None, svd_type=0):
        """
        Perform MCA analysis using singular value decomposition of the Burt matrix
        About 10000 points is good for local computer
        Automatically removes columns from analysis if not present in sample passed in
        :param data: input data features expanded as binary dummy values
        :param K: number of total features (before dummy variable expansion)
        :param benzecri: optional correction to eigenvalues and explained variance (optimistic estimation)
        :param greenacre: optional correction to eigenvalues and explained variance
        """
        print('Fitting with MCA...')

        # # Pandas
        # # Remove columns of all zeros (assume all features have at least one nonzero entry)
        # table = data
        # table = table.loc[:, (table != 0).any(axis=0)]
        # self.removed_cols = list(sorted(set(data).difference(set(table))))
        # print('Columns not found in sample:')
        # print(self.removed_cols, '\n')
        # self.categories = list(table)
        # self.m = table.shape[0]
        # self.J = table.shape[1]
        # values = table.values

        # In case it takes too much memory to do SVD for the entire matrix
        if self.n_samples < data.shape[0]:
            values = data[:self.n_samples, :]
        else:
            values = data
        # values = data

        # Remove columns of all zeros (assume all features have at least one nonzero entry)
        ind = (np.sum(values, axis=0) != 0)
        self.keep_columns = np.ravel(ind)
        values = values[:, self.keep_columns]
        self.m = values.shape[0]
        self.J = values.shape[1]
        print('number of dropped columns: ', data.shape[1] - np.count_nonzero(self.keep_columns))

        # Normalize
        N = np.sum(values)
        Z = values/N

        # Compute row and column sums
        sum_r = np.ravel(np.sum(Z, axis=1))
        sum_c = np.ravel(np.sum(Z, axis=0))

        # Compute residual
        rcT = np.outer(sum_r, sum_c)
        Z_residual = Z - rcT
        # Z_residual = Z

        # Scale residual by the square root of column and row sums.
        # Note we are computing SVD on residual matrix, not the analogous covariance matrix,
        # Therefore, we are dividing by square root of Sums.
        sum_r_sqrt = sum_r**(-0.5)
        sum_c_sqrt = sum_c**(-0.5)

        # Dr_Z = np.dot(D_r_sqrt_mi, Z_residual)
        # MCA_mat = np.dot(Dr_Z, D_c_sqrt_mi)
        Dr_Z = np.multiply(sum_r_sqrt[:, None], Z_residual)
        MCA_mat = np.multiply(Dr_Z, sum_c_sqrt)

        # Apply SVD.
        # IN np implementation, MCA_mat = U*S*V_T, not U*S*V_T'
        U, S, V_T = scipy.sparse.linalg.svds(MCA_mat, k=(min(MCA_mat.shape)-1))

        # Sort by eigenvalue in descending order
        idx = np.argsort(-S)
        S = S[idx]
        U = U[:, idx]
        V_T = V_T[idx, :]

        V = np.transpose(V_T)
        self.S = S

        # Standard coordinates
        A = np.multiply(sum_r_sqrt[:, None], U)  # Rows
        B = np.multiply(sum_c_sqrt[:, None], V)  # Columns

        # Principal coordinates (omitting for rows since storing would take a lot of memory)
        self.G = np.multiply(B[:, :S.shape[0]], S)  # Columns

        # Get eigenvalues (square of Burt matrix eigenvalues)
        Lam = S**2

        # Apply optional corrections to eigenvalues and explained variance
        if self.correction=='benzecri':
            self.E, self.expl_var = self.benzecri_correction(Lam)
        elif self.correction=='greenacre':
            self.E, self.expl_var = self.greenacre_correction(Lam)
        else:
            self.E = Lam
            self.expl_var = Lam/np.sum(Lam)

        self.k = next(i for i, x in enumerate(np.cumsum(self.expl_var)) if x > (self.var_cutoff/100))
        self.tot_expl_var = np.cumsum(self.expl_var)[self.k-1]
        print('Dimension of truncated basis (%.2f %% explained variance):' % self.tot_expl_var, self.k, '\n')

        return self

    def benzecri_correction(self, eig_val):
        """
        Calculate Benzecri corrected eigenvalues and corrected explained variance for each eigensolution
        :param eig_val: eigenvalues of data
        :return: eigenvalues, explained variance
        """
        E = np.array([(self.K / (self.K - 1.)**2 * (lm - 1. / self.K))**2
                      if (lm > 1. / self.K) else 0 for lm in eig_val])
        expl_var_bn = E / np.sum(E)
        return E, expl_var_bn

    def greenacre_correction(self, eig_val):
        """
        Calculate Greenacre corrected explained variance for each eigensolution
        :param eig_val: eigenvalues of data
        :return: Benzecri corrected eigenvalues, explained variance
        """
        E = np.array([(self.K / (self.K - 1.)**2 * (lm - 1. / self.K))**2
                      if (lm > 1. / self.K) else 0 for lm in eig_val])
        avg_inertia = (self.K / (self.K - 1.) * (np.sum(eig_val**2) - (self.J - self.K) / self.K**2.))
        expl_var_gn = E / avg_inertia
        return E, expl_var_gn

    def squared_cosine_distance(self, X, transformed=True):
        """
        Squared cosine distance between individuals / features and truncated basis vectors
        :return: for (individuals, features)
        """
        if not transformed:
            F = self.transform(X)
        else:
            F = X
        d_r = np.linalg.norm(F[:, :self.k], axis=1)**2
        d_c = np.linalg.norm(self.G[:, :self.k], axis=1)**2
        cosdist_r = np.apply_along_axis(lambda x: x/d_r, 0, F[:, :self.k]**2)
        cosdist_c = np.apply_along_axis(lambda x: x/d_c, 0, self.G[:, :self.k]**2)
        return cosdist_r, cosdist_c

    def contributions(self, X, transformed=True):
        """
        Contributions between between individuals / features and truncated basis vectors
        :param transformed: whether the input data, X, has already been transformed
        :return: for (individuals, features)
        """
        if not transformed:
            F = self.transform(X)
        else:
            F = X
        cont_r = np.apply_along_axis(lambda x: x/self.S[:self.k]**2/self.m, 1, F[:, :self.k] ** 2)
        cont_c = np.apply_along_axis(lambda x: x/self.S[:self.k]**2/self.m, 1, self.G[:, :self.k] ** 2)
        return cont_r, cont_c

    def transform(self, data):
        """
        Project data into the reduced factor space
        :param data: numpy array of data represented using original features
        :return: numpy array of data represented using reduced factor space
        """
        X = data[:, self.keep_columns]
        return np.dot(X, self.G[:, :self.k]) / self.S[:self.k] / self.K

    def scree_plot(self):
        """
        Plot Scree plot for convergence of explained variance w.r.t eigenvectors
        """
        cum_expl_var = np.cumsum(self.expl_var)
        plt.plot(self.expl_var * 100, 'k.-', label="variance")
        plt.plot(cum_expl_var * 100, 'b.-', label="cumulative variance")
        plt.axvline(x=(self.k + 0.5), color='r', linestyle='--')
        x1 = (self.k + 0.5) + (plt.xlim()[1] - plt.xlim()[0]) * 0.02
        y1 = plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.8
        x2 = (self.k + 0.5) + (plt.xlim()[1] - plt.xlim()[0]) * 0.02
        y2 = plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.75
        plt.text(x1, y1, '%.2f %% explained variance' % self.tot_expl_var, color='r')
        plt.text(x2, y2, '%.f vectors' % self.k, color='black', alpha=0.8)
        plt.xlabel('eigensolution')
        plt.ylabel('percentage')
        plt.title('MCA Scree Plot')
        plt.legend()
        plt.show()

    def plot_data(self, data):
        """
        Plot data points (rows) as component of top two basis vectors
        """
        data_proj = self.transform(data)
        x = data_proj[:, 0]
        y = data_proj[:, 1]
        plt.scatter(x, y, alpha=0.5)
        plt.axhline(0, linestyle='--', color='k', alpha=0.5)
        plt.axvline(0, linestyle='--', color='k', alpha=0.5)
        plt.xlabel('dim1 ({0}%)'.format(int(self.expl_var[0] * 100)))
        plt.ylabel('dim2 ({0}%)'.format(int(self.expl_var[1] * 100)))
        plt.title('MCA: data on first two components')
        plt.show()

    def symmetric_map_plot(self, X, transformed=True, categories=None, labels=False, toplot='both', colors=None):
        """
        Plot symmetric map of data points on first two principle components
        :param X: data to be plotted
        :param transformed: whether the input data, X, has already been transformed
        :param categories: list of categories returned by DictVectorizer.get_feature_names()
        :param labels: boolean determining whether or not to print category names
        :param toplot: which coordinates to plot
        :param colors: what metric to use for colors
        """
        if toplot not in ['both', 'rows', 'cols']:
            raise ValueError('option toplot must be both, rows, or cols.')
        if colors is not None and colors != 'cosdist' and colors != 'contr':
            raise ValueError('option colors must be None, cosdist, or contr.')

        if not transformed:
            F = self.transform(X)
        else:
            F = X

        # Add points for rows
        if toplot == 'both' or toplot == 'rows':
            x_rows = F[:, 0]
            y_rows = F[:, 1]
            if colors == 'cosdist':
                cvar = np.sum(self.squared_cosine_distance(F)[0][:, :2], axis=1)
                label = 'squared cosine distance'
                sc = plt.scatter(x_rows, y_rows, alpha=0.7, c=cvar, cmap='viridis', marker='o')
                cbar = plt.colorbar(sc)
                cbar.set_label(label, rotation=90, labelpad=10)
            else:
                plt.scatter(x_rows, y_rows, alpha=0.5, color='b', marker='o')

        # Add points for columns
        if toplot == 'both' or toplot == 'cols':
            x_cols = self.G[:, 0]
            y_cols = self.G[:, 1]
            if colors == 'contr':
                cvar = np.sum(self.contributions(F)[1][:, :2], axis=1)
                print(cvar.shape)
                label = 'contributions'
                sc = plt.scatter(x_cols, y_cols, alpha=0.7, c=cvar, cmap='inferno', marker='^')
                cbar = plt.colorbar(sc)
                cbar.set_label(label, rotation=90, labelpad=10)
            else:
                plt.scatter(x_cols, y_cols, alpha=0.7, color='r', marker='^')
            if labels:
                for i, txt in enumerate(categories):
                    plt.annotate(txt, (x_cols[i], y_cols[i]))

        # Add global plot features
        plt.axhline(0, linestyle='--', color='k', alpha=0.5)
        plt.axvline(0, linestyle='--', color='k', alpha=0.5)
        plt.xlabel('dim1 ({0}%)'.format(int(self.expl_var[0] * 100)))
        plt.ylabel('dim2 ({0}%)'.format(int(self.expl_var[1] * 100)))
        plt.title('MCA: Symmetric Map')
        plt.show()

    def correlation_plot(self, X, transformed=True, categories=None, norm='dim'):
        """
        Print correlation plot showing which features contribute to reduced basis vectors
        :param X: data to be plotted
        :param transformed: whether the input data, X, has already been transformed
        :param categories: list of categories returned by DictVectorizer.get_feature_names()
        :param norm: which axis to normalize, default is by basis dimension
        """
        def set_size(w, h, ax=None):
            """ w, h: width, height in inches """
            if not ax:
                ax = plt.gca()
            l = ax.figure.subplotpars.left
            r = ax.figure.subplotpars.right
            t = ax.figure.subplotpars.top
            b = ax.figure.subplotpars.bottom
            figw = float(w) / (r - l)
            figh = float(h) / (t - b)
            ax.figure.set_size_inches(figw, figh)

        xyMat = self.contributions(X, transformed=transformed)[1]
        if norm=='cat':
            sum_r = np.sum(xyMat, axis=1)
            xyMat = np.apply_along_axis(lambda x: x / sum_r, 0, xyMat)
        basis_labels = ['dim' + str(i+1) for i in range(self.k)]
        xdim = xyMat.shape[1]
        ydim = xyMat.shape[0]
        x = np.arange(xdim)
        y = np.arange(ydim)

        xvec = []
        yvec = []
        cvec = []
        scfac = np.average(xyMat)*1e4
        for r in range(ydim):
            for c in range(xdim):
                cVal = xyMat[r, c]
                cvec.append(cVal * scfac)
                xvec.append(c)
                yvec.append(r)

        unit = 0.8
        fig, ax = plt.subplots()

        ax.scatter(xvec, yvec, c='g', s=cvec, alpha=0.6)
        ax.grid(color='0.5')
        ax.set_xlim(-1, xdim + 1)
        ax.set_ylim(-1, ydim + 1)
        ax.set_xticks(x)
        ax.set_yticks(y)
        ax.set_xticklabels(basis_labels, rotation='vertical', multialignment='center', size=10)
        ax.set_yticklabels(categories, rotation=45, multialignment='right', size=10)
        set_size(unit*xdim, unit*ydim/2, ax=ax)
        plt.show()