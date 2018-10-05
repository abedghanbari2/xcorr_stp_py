import numpy as np
from scipy import io
import scipy.signal as sps
from scipy.optimize import minimize
import time
from patsy import dmatrix
import bisect
import matplotlib.pyplot as plt
import statsmodels.api as sm
import cython_loop
import lam_cython
import math

class SpikeAnalysis:

    def __init__(self, st1, st2, ta=-.1, tb=.1, nbins=100):
        self.st1 = st1
        self.st2 = st2
        self.ta = ta
        self.tb = tb
        self.nbins = nbins
        self.thr = .1

    def cross_correlogram(self):
        all = []
        for ref in self.st1:
            all.extend(list(self.st2[(self.st2 > ref + self.ta) & (self.st2 < ref + self.tb)] - ref))
        return all

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def logit(x):
        return np.log(x / (1 - x))

    def syn(self, xx):
        t_delay = self.tb * SpikeAnalysis.sigmoid(xx[0])
        tau = .01 * SpikeAnalysis.sigmoid(xx[1])
        return lambda t: xx[2] * max(0, t - t_delay) / tau * np.exp(1 - max(0, t - t_delay) / tau)

    def func_min_xcorr(self, x, bas, y):
        alpha = self.syn(x[-3:])
        synapse = [alpha(i) for i in np.linspace(self.ta, self.tb, self.nbins)]
        lam = np.exp(x[:-3].dot(bas) + synapse)
        return np.sum(-np.multiply(x[:-3].dot(bas) + synapse, y)) + np.sum(lam)

    def estimate_synapse(self, num_basis=5, num_rept=100, rnd_scale=1, plot_flag=0):
        """
        estimate synapse from cross-correlogram accounting for slow fluctuations in xcorr
        """
        xcorr1 = np.histogram(self.cross_correlogram(), self.nbins)
        xx = np.linspace(1, num_basis - 1, self.nbins)
        bas = np.zeros(shape=(num_basis + 1, self.nbins))
        for i in range(num_basis + 1):
            bas[i] = sps.bspline(xx - i, 3).flatten()
        # warm-up
        func_val = float('inf')
        for i in range(num_rept):
            x0 = np.random.randn(1, np.shape(bas)[0] + 3) * rnd_scale
            res_temp = minimize(self.func_min_xcorr, x0, method="l-bfgs-b", args=(bas, xcorr1[0]),
                                options={'maxfun': 150, 'maxiter': 150})
            if res_temp.fun < func_val:
                func_val = res_temp.fun
                res_warmup = res_temp
        res = minimize(self.func_min_xcorr, res_warmup.x, method="l-bfgs-b", args=(bas, xcorr1[0]),
                       options={'maxfun': 15000, 'maxiter': 15000})
        t_xcorr = np.linspace(self.ta, self.tb, self.nbins)
        alpha = self.syn(res.x[-3:])
        synapse = [alpha(i) for i in t_xcorr]  # the synapse
        slow_xcorr = res.x[:-3].dot(bas)
        if plot_flag == 1:
            plt.bar(t_xcorr, xcorr1[0], width=t_xcorr[1] - t_xcorr[0])
            plt.plot(t_xcorr, np.exp(slow_xcorr), color='r', linewidth=3)  # the smooth CommInp
            plt.plot(t_xcorr, np.exp(slow_xcorr + synapse), color='k', linewidth=3)
            plt.show()
        return res, synapse, alpha, slow_xcorr, xcorr1[0], t_xcorr

    @staticmethod
    def isi_tlist(spk):
        isi_pre = np.diff(spk.ravel())
        isi_pre = np.append(np.max(isi_pre), isi_pre)
        return isi_pre

    def time_post_pre(self):
        return [self.st2[bisect.bisect_left(self.st2, i) - 1] - i for i in self.st1]

    def time_pre_cpl(self):
        len_st2 = len(self.st2)
        return np.array([self.st2[bisect.bisect_left(self.st2, i)] - i if bisect.bisect_left(self.st2, i) < len_st2 else None for i in self.st1])

    def y_mat(self, t_pre_cpl, num_bins=100):
        """
        Creates a matrix each row representing a presynaptic spike
        0/1 values in each column shows presence of postsynaptic spike

        Input:
            t_pre_cpl : time difference between each presynaptic spike and following postsynaptic spike
            num_bins : number of bins for Y matrix // should be the same as columns in X

        """
        Ycpl_idx = np.array([np.digitize(i, np.linspace(0, self.tb, num_bins + 1)) for i in t_pre_cpl])
        idx = np.hstack((np.array([list(range(0, len(self.st1)))]).T, np.reshape(Ycpl_idx, (-1, 1))))
        Ycpl = np.zeros([len(self.st1), num_bins + 1])
        for i in range(len(self.st1)):
            Ycpl[idx[i, 0], idx[i, 1] - 1] = 1  # why do i have -1 here?
        return Ycpl[:, :-1]

    @staticmethod
    def mask_Yy(Yy):
        mYy = Yy.copy()
        for i in range(Yy.shape[0]):
            for j in range(Yy.shape[1]):
                if Yy[i, j] == 1:
                    break
                else:
                    mYy[i, j] = 1
        return mYy

    def make_X(self, syn_param, max_history_filter=.01, num_history_splines=10, len_fr_bas=200, plot_h_flag=0):
        t_post_pre = self.time_post_pre()
        x = np.negative(t_post_pre)
        x[x < 0] = np.median(x[x > 0])  # just a check but be careful
        knots_history = np.exp(np.linspace(np.log(min(x) + .0001), np.log(max_history_filter - .0001), num_history_splines))
        X_h = dmatrix("bs(x, degree=3, knots = knots_history, include_intercept=False) - 1", {"x": x})
        if plot_h_flag == 1:
            for i in range(X_h.shape[1] - 4):
                plt.scatter(x, X_h[:, i])
            plt.xlabel('pre spike time - previous postsynaptic spike')
            plt.xlim([0, max_history_filter])
            plt.show()
        X_h_truncated = X_h[:, : num_history_splines - 4]  # don't need them
        # baseline firing rate covariates
        knots_fr = np.linspace(np.min(self.st1) + 1, np.max(self.st1), int((np.floor(max(self.st1))) / len_fr_bas))
        X_fr = dmatrix("bs(x, degree=3, knots = knots_fr, include_intercept=False) - 1", {"x": self.st1})
        X_fr_truncated = X_fr[:, : -1]

#        Xvar = np.hstack((np.array(X_fr_truncated)))
        Xvar = np.hstack((np.array(X_fr_truncated), np.array(X_h_truncated)))
#        Xvar = np.hstack((np.ones([len(self.st1), 1]), np.array(X_fr_truncated), np.array(X_h_truncated)))
#        X_notNone = Xvar[[i for i in range(len(Xvar[:, -1])) if Xvar[:, -1][i] is not None], :].astype('float')
        return Xvar

    @staticmethod
    def transf_theta(theta):
        """
        transforms the parmaters of the stp model so we don;t have to use
        constrained optimization
        """
#        theta_ = [5 * SpikeAnalysis.sigmoid(theta[-5]),
#              5 * SpikeAnalysis.sigmoid(theta[-4]),
#              SpikeAnalysis.sigmoid(theta[-3]),
#              SpikeAnalysis.sigmoid(theta[-2]),
#              .2 * SpikeAnalysis.sigmoid(theta[-1]) ]
        theta_ = [np.exp(theta[-5]),
                  np.exp(theta[-4]),
                  SpikeAnalysis.sigmoid(theta[-3]),
                  SpikeAnalysis.sigmoid(theta[-2]),
                  np.exp(theta[-1])]
        return theta_

    def alpha_mat(self, synapse):
        return np.matlib.repmat(synapse[100:], self.st1.shape[0], 1)

    def lambda_fun(self, beta, X, psp_scaled, synapse, mY):
#        dyn_stp_mat = (np.array([psp_scaled]) * np.array([synapse[100:]]).T).T
#        lam = np.multiply(SpikeAnalysis.sigmoid(np.array([np.dot(X, beta)]).T + dyn_stp_mat), mY)

        # cythonized x2 speed
        lam = np.array(lam_cython.lambda_fun_cython(beta, X, psp_scaled, np.array(synapse), mY))
        return lam

    @staticmethod
    def log_likelihood(x, lam_mat, Y):
        logl0 = np.zeros(lam_mat.shape)
        logl0[lam_mat == 0] = 1
        logl1 = np.zeros(lam_mat.shape)
        logl1[lam_mat == 1] = 1
        return -np.sum(np.multiply(np.log(lam_mat + logl0), Y) + np.multiply(np.log(1 - lam_mat + logl1), 1 - Y)) + 1 * np.sum(np.array([x[i] for i in [-6, -5, -2]])**2)

    def cost_func_and_jac(self, x, X, synapse, Y, mY, jac_flag=False):
        theta_ = SpikeAnalysis.transf_theta(x[-6:-1])
        psp = self.stp_model(theta_)
        lam_mat = self.lambda_fun(x[:-6], X, x[-1] * psp, synapse, mY)
        jac_vec = np.zeros(x.shape)
        if jac_flag is True:
            jac_vec[0:(len(x) - 6)] = np.sum(np.dot(X.T, lam_mat - Y), axis=1)
            for i in range(1, len(jac_vec)):
                if i >= len(x) - 6:
                    dx = np.zeros(x.shape)
                    eps = .001
#                    eps = x[i]/10*np.random.rand(1)
                    dx[i] = eps
                    jac_vec[i] = (self.cost_func_and_jac(x + dx, X, synapse, Y, mY, False)[0] - self.cost_func_and_jac(x - dx, X, synapse, Y, mY, False)[0]) / 2 / eps

#        print('jac estimated: ', np.around(jac_vec[-6:], 2))
        if np.random.rand(1)<.05:
            print('function val: ', np.around(SpikeAnalysis.log_likelihood(x, lam_mat, Y), 2))
            print('stp params: ', np.around(SpikeAnalysis.transf_theta(x[-6:-1]), 2))
            
            plt.scatter(np.negative(self.time_post_pre()), SpikeAnalysis.sigmoid(X[:,-9:].dot(x[-15:-6])))
            plt.xlim([0,.01])
            plt.show()
                        
            plt.scatter(self.st1[:30],psp[:30])
            plt.show()
#            print('fun value: ',self.log_likelihood(x, lam_mat, Y))
#            self.spike_trans_prob_est(x, X, synapse, N=100, plot_flag=1)
#            t_xcorr = np.linspace(self.ta, self.tb, self.nbins)
#            t_syn_interval = t_xcorr[synapse>.1*np.max(synapse)]
#            self.spike_transmission_prob(np.min(t_syn_interval), np.max(t_syn_interval), num_isilog=50, plot_flag=1)
#            plt.show()

        return SpikeAnalysis.log_likelihood(x, lam_mat, Y), jac_vec

    def optim_func(self, X, synapse, Y, rnd_scale):
        """
        optimizing the cost function
            - initialization with standard glm % makes it more unstable ...
            - warm start from random restart
        """
        method_opt = "cg"
        mY = self.mask_Yy(Y)
        t0 = time.time()

        # initialization with the help of GLM
        binomial_model = sm.GLM(np.sum(Y, axis=1), X, family=sm.families.Binomial())
        binomial_results = binomial_model.fit()

        # warm start
        func_val = 1e12
        for i in range(5):
            x0 = np.random.randn(np.shape(X)[1] + 6, 1) * rnd_scale
            x0[-6:-1] = x0[-6:-1] + np.array([-1, -1, 0, 0, -3], ndmin=2).T
            x0[:-6] = x0[:-6] + np.reshape(binomial_results.params, (-1, 1))
            res = minimize(self.cost_func_and_jac, x0, method=method_opt, args=(X, synapse, Y, mY, True), jac=True, options={'maxfun': 15, 'maxiter': 5})

            if res.fun < func_val:
                func_val = res.fun
                res_warmup = res

        print('finished initialization and warm-start in:', time.time() - t0)
        res_final = minimize(self.cost_func_and_jac, res_warmup.x, method=method_opt, args=(X, synapse, Y, mY, True), jac=True,
                             options={'factr': 10, 'maxfun': 15000, 'maxiter': 15000})
        return res_final

    def stp_model(self, theta):
        """
        cython implementation of the Tsodyks and Markram model of a dynamical synapse
        """
        isi = SpikeAnalysis.isi_tlist(self.st1)
        psp = cython_loop.stp_model_cython(isi, np.array(theta))
        psp[0] = np.median(psp)
        psp = psp / np.mean(psp)
        if math.isnan(np.sum(psp)):
            raise ValueError("psp has nan values ...")
        return psp

    def spike_transmission_prob(self, min_syn=.0005, max_syn=.0035, num_isilog=50, plot_flag=0):
        """
        plot the spike tranmission probability from pre- and postsynaptic spikes
        """
        isi_pre = SpikeAnalysis.isi_tlist(self.st1)
        # a log-spaced vector [lowest ISI, highest ISI] presynaptic ISI
        logisi_vec = np.logspace(np.log10(np.min(isi_pre)), np.log10(np.max(isi_pre)), base=10.0, num=num_isilog)
#         for each presynaptic spike what was the ISI before and build a post_spk_presence (0/1)
#         in syn_interval after that presynaptic spike
        post_spk_presence = np.array([1 if np.any((self.st2 > i + min_syn) & (self.st2 < i + max_syn)) else 0 for i in self.st1])
        isi_split_num = np.digitize(isi_pre, logisi_vec)
        spk_prob = np.zeros(len(logisi_vec))
        t_split_isi = np.zeros(len(logisi_vec))
        for j in range(len(logisi_vec)):
            spk_prob[j] = np.mean(post_spk_presence[isi_split_num == j])
            t_split_isi[j] = np.mean(isi_pre[isi_split_num == j])
        if plot_flag == 1:
            plt.scatter(np.log10(t_split_isi), spk_prob)
#            plt.show()
        return spk_prob, t_split_isi

    @staticmethod
    def cumulitive_lam(lam_mat, idx):
        """
        cumulitive probability of firing from binned lambdas
        """
#        print(lam_mat.shape)
        lam_cum = np.zeros([lam_mat.shape[0], 1])
        lam_chunk = lam_mat[:, idx]
        anti_lam = 1 - lam_chunk
        for ispk in range(lam_chunk.shape[0]):
            lam_cum_each = lam_chunk[ispk, 0]
            for i in range(1, lam_chunk.shape[1]):
                lam_cum_each += np.prod(anti_lam[ispk, :i]) * lam_chunk[ispk, i]
            lam_cum[ispk] = lam_cum_each
        return lam_cum

    def spike_trans_prob_est(self, x, X, synapse, N=40, plot_flag=0):
        """
        spike tranmission probability from lambda with estimated parameters
        """
        psp = self.stp_model(SpikeAnalysis.transf_theta(x[-6:-1]))
        lam = self.lambda_fun(x[:-6], X, x[-1] * psp, synapse, np.ones([X.shape[0],int(len(synapse)/2)]))
        idx_syn = [i - 100 for i in range(len(synapse)) if synapse[i] > self.thr * np.max(synapse)]
#        eff_n = np.sum(lam[:, idx_syn], axis=1)
        eff_n = SpikeAnalysis.cumulitive_lam(lam, idx_syn)
        isilog10 = np.log10(SpikeAnalysis.isi_tlist(self.st1))
        isi_vec = np.linspace(min(isilog10) - .0001, max(isilog10) + .0001, N)
        idx_isi = np.array([int(np.digitize(i, isi_vec)) for i in isilog10])
        mean_eff = np.zeros([N, 1])
        for i in range(N):
            mean_eff[i] = np.mean(eff_n[idx_isi == i])
        if plot_flag == 1:
            plt.scatter(isi_vec, mean_eff)
        return mean_eff, isi_vec, eff_n

    @staticmethod
    def load_data(i):
        if i == 1:

            # LGN - CTX
            mat = io.loadmat('/Volumes/GoogleDrive/My Drive/stp in vivo/data/swadlow/July13A.mat')
            pre = mat['LGN'][:-1]
            post = mat['CTX']

        elif i == 2:

            # AVCN
            mat = io.loadmat('/Volumes/GoogleDrive/My Drive/stp in vivo/data/avcn/Tuning_G1003C82R14.mat')
#            mat = io.loadmat('C:\\Users\\abg14006\\Google Drive\\stp in vivo\\data\\data_remote\\Tuning_G1003C82R14.mat')

            pre = mat['Tuning']['Raw'][0][0]['Continuous'][0][0]['STsEPSP'][0][0] / mat['Tuning']['SamplingRate'][0][0]
            temp_a = mat['Tuning']['Raw'][0][0]['Continuous'][0][0]['STs'][0][0]
            temp_b = mat['Tuning']['Raw'][0][0]['Continuous'][0][0]['isAP'][0][0]
            post = [val for i, val in enumerate(temp_a) if temp_b[i] == 1] / mat['Tuning']['SamplingRate'][0][0]

        elif i == 3:

            # VB-Barrel
            mat = io.loadmat('/Volumes/GoogleDrive/My Drive/stp in vivo/data/data_remote/Mar23c1,2,4_Herc_spikes.mat')
#            mat = io.loadmat('C:\\Users\\abg14006\\Google Drive\\stp in vivo\\data\\data_remote\\Mar23c1,2,4_Herc_spikes.mat')

            pre = mat['Tlist'][0][0] + .0001 * np.random.random_sample(mat['Tlist'][0][0].shape)
            post = mat['Tlist'][0][1] + .0001 * np.random.random_sample(mat['Tlist'][0][1].shape)
            t_start = np.floor(np.min(np.append(pre, post)))
            pre = pre - t_start
            post = post - t_start

        return pre, post


    #%% main code
if __name__ == "__main__":
    pass
