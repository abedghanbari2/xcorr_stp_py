import numpy as np
from scipy import io
import scipy.signal as sps
# import scipy.interpolate as spi
from scipy.optimize import minimize
import time
from patsy import dmatrix
import bisect
import matplotlib.pyplot as plt


class SpikeAnalysis:

    def __init__(self, st1, st2, ta=-.1, tb=.1, nbins=100):
        self.st1 = st1
        self.st2 = st2
        self.ta = ta
        self.tb = tb
        self.nbins = nbins

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
        t_delay = self.tb * self.sigmoid(xx[0])
        tau = .01 * self.sigmoid(xx[1])
        return lambda t: xx[2] * max(0, t - t_delay) / tau * np.exp(1 - max(0, t - t_delay) / tau)

    def func_min_xcorr(self, x, bas, y):
        alpha = self.syn(x[-3:])
        synapse = [alpha(i) for i in np.linspace(self.ta, self.tb, self.nbins)]
        lam = np.exp(x[:-3].dot(bas) + synapse)
        return np.sum(-np.multiply(x[:-3].dot(bas) + synapse, y)) + np.sum(lam)

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
        idx = np.hstack((np.array([list(range(0, len(self.st1)))]).T, np.reshape(Ycpl_idx,(-1,1))))
        Ycpl = np.zeros([len(self.st1), num_bins + 1])
        for i in range(len(self.st1)):
            Ycpl[idx[i, 0], idx[i, 1] - 1] = 1 #why do i have -1 here?
        return Ycpl[: , :-1]

    def make_X(self, syn_param, max_history_filter=.01, num_history_splines=10, len_fr_bas=200, plot_h_flag=0):
        t_post_pre = self.time_post_pre()
        x = np.negative(t_post_pre)
        x[x < 0] = np.median(x[x > 0]) # just a check but be careful 
        knots_history = np.linspace(min(x) + .0001, max_history_filter - .0001, num_history_splines)
        X_h = dmatrix("bs(x, degree=3, knots = knots_history, include_intercept=False) - 1", {"x": x})
        if plot_h_flag == 1:
            for i in range(X_h.shape[1] - 4):
                plt.scatter(x, X_h[:, i])
            plt.xlabel('pre spike time - previous postsynaptic spike')
            plt.xlim([0, max_history_filter])
            plt.show()
        X_h_truncated = X_h[: , : num_history_splines - 4] # don't need them
        # baseline firing rate covariates
        knots_fr = np.linspace(np.min(self.st1) + 1, np.max(self.st1), int((np.floor(max(self.st1))) / len_fr_bas))
        X_fr = dmatrix("bs(x, degree=3, knots = knots_fr, include_intercept=False) - 1", {"x": self.st1})
        X_fr_truncated = X_fr[: , : -1]
        Xvar = np.hstack((np.ones([len(self.st1), 1]), np.array(X_fr_truncated), np.array(X_h_truncated)))
#        X_notNone = Xvar[[i for i in range(len(Xvar[:, -1])) if Xvar[:, -1][i] is not None], :].astype('float')
        return Xvar

    def lambda_fun(self, beta, X, psp_scaled, synapse):
        Xmat = np.matlib.repmat(np.array([np.dot(X, beta)]).T, 1, 100)
        alph_mat = np.matlib.repmat(synapse[100:], X.shape[0], 1)
        for i in range(len(psp_scaled)):
            alph_mat[i, : ] = psp_scaled[i] * alph_mat[i, : ]
        return self.sigmoid(Xmat + alph_mat)

    @staticmethod
    def log_likelihood(lam_mat, Y):
        return -np.sum(np.multiply(lam_mat, Y) + np.multiply(1 - lam_mat, 1 - Y))

    def transf_theta(self,x):
        """
        transforms the parmaters of the stp model so we don;t have to use 
        constrained optimization
        """
#        x_ = [5 * self.sigmoid(x[-5]), 
#              5 * self.sigmoid(x[-4]), 
#              self.sigmoid(x[-3]), 
#              self.sigmoid(x[-2]),
#              .2 * self.sigmoid(x[-1]) ]
        x_ = [np.exp(x[-5]), 
              np.exp(x[-4]), 
              self.sigmoid(x[-3]), 
              self.sigmoid(x[-2]),
              np.exp(x[-1]) ]
        return x_
    
    def cost_func(self, x, X, synapse, Y):
        theta_ = self.transf_theta(x[-6:-1])
        psp = self.stp_model(theta_)
        lam_mat=self.lambda_fun(x[:-6], X, x[-1] * psp, synapse)
#        lam_mat=self.lambda_fun(x[:-6], X, x[-1] * np.ones([X.shape[0],1]), synapse)
#        print(self.log_likelihood(lam_mat, Y))
        if np.random.rand(1) < .1:
            #            print(5 * self.sigmoid(x[-6]), 5 * self.sigmoid(x[-5]), self.sigmoid(x[-4]), self.sigmoid(x[-3]), .2 * self.sigmoid(x[-2]))
#            plt.plot(np.sum(Y,axis=1)/np.max(lam_mat))
            plt.plot(self.sigmoid(np.dot(X, x[:-6])))
            plt.show()
#        print(x[-6:-1])
        return self.log_likelihood(lam_mat, Y)

    def cost_func_jac(self, x, X, synapse, Y):
        theta_ = self.transf_theta(x[-6:-1])
        psp = self.stp_model(theta_)
        lam_mat=self.lambda_fun(x[:-6], X, x[-1] * psp, synapse)
        eps=.00001
        beta=x[:-6]
        jac_vec=np.zeros(x.shape)
        for i in range(len(jac_vec)):
            if i < len(beta):
                jac_vec[i]=np.sum(np.dot(np.array([X[:, i]]), Y - lam_mat))
            else:
                dx=np.zeros(x.shape)
                dx[i]=eps
                jac_vec[i]=(self.cost_func(x + dx, X, synapse, Y) - self.cost_func(x - dx, X, synapse, Y)) / 2 / eps

        return jac_vec

    def optim_func(self, X, synapse, Y, rnd_scale):
        x0=np.random.randn(np.shape(X)[1] + 6, 1) * rnd_scale
        res=minimize(self.cost_func, x0, method = "l-bfgs-b", args = (X, synapse, Y),
                       options={'maxfun': 1500, 'maxiter': 1500})
#        stp_param = [ 5*self.sigmoid(x[-6]), 5*self.sigmoid(x[-5]), self.sigmoid(x[-4]), self.sigmoid(x[-3]), np.exp(x[-2]) ]
        return res

    def stp_model(self, theta):
        tauD, tauF, U, f, tauInteg = theta
        isi = self.isi_tlist(self.st1)
        exp_dep = np.exp(-isi / tauD)
        exp_fac = np.exp(-isi / tauF)
        exp_integ = np.exp(-isi / tauInteg)

        R = np.zeros([len(isi), 1])
        u = np.zeros([len(isi), 1])
        integ = np.zeros([len(isi), 1])
        psp = np.zeros([len(isi), 1])
        R[0] = 1
        u[0] = U
        psp[0] = U
        L = len(isi) - 1
        for i in range(L):
            R[i + 1] = 1 - (1 - R[i] * (1 - u[i])) * exp_dep[i]
            u[i + 1] = U + (u[i] + f * (1 - u[i]) - U) * exp_fac[i]
            integ[i + 1] = psp[i] * exp_integ[i]

            psp[i + 1] = integ[i + 1] + R[i + 1] * u[i + 1]
        return psp

    def estimate_synapse(self, num_basis=5, num_rept=100, rnd_scale=1, plot_flag=0):
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
            plt.bar(t_xcorr, xcorr1[0], width=t_xcorr[1]-t_xcorr[0])
            plt.plot(t_xcorr, np.exp(slow_xcorr),color='r',linewidth=3) # the smooth CommInp
            plt.plot(t_xcorr, np.exp(slow_xcorr+synapse),color='k',linewidth=3)
            plt.show()
        return res, synapse, alpha, slow_xcorr, xcorr1[0], t_xcorr

    def spike_transmission_prob(self, min_syn=.0005, max_syn=.0035, num_isilog=50):
        # calculates efficacy
        isi_pre = self.isi_tlist(self.st1)
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
        return spk_prob, t_split_isi

    def spike_trans_prob_est(self, x, X, synapse, N=40, plot_flag=0):
        """
        plot the spike tranmission probability from estimated parameters
        """
        psp = self.stp_model(self.transf_theta(x[-6:-1]))
        lam = self.lambda_fun(x[:-6], X, x[-1]*psp, synapse)
        
        idx_syn = [i-100 for i in range(len(synapse)) if synapse[i]>.1*np.max(synapse)]
        eff_n = np.sum(lam[:,idx_syn],axis=1)
        isilog10 = np.log10(self.isi_tlist(self.st1))
        isi_vec = np.linspace(min(isilog10)-.0001,max(isilog10)+.0001,N)
        idx_isi = np.array([int(np.digitize(i,isi_vec)) for i in isilog10])
        mean_eff = np.zeros([N,1])
        for i in range(N):
            mean_eff[i] = np.median(eff_n[idx_isi==i])
        if plot_flag==1:
            plt.plot(isi_vec,mean_eff)
            plt.show()
        return mean_eff, isi_vec

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
            pre = mat['Tuning']['Raw'][0][0]['Continuous'][0][0]['STsEPSP'][0][0] / mat['Tuning']['SamplingRate'][0][0]
            temp_a = mat['Tuning']['Raw'][0][0]['Continuous'][0][0]['STs'][0][0]
            temp_b = mat['Tuning']['Raw'][0][0]['Continuous'][0][0]['isAP'][0][0]
            post = [val for i, val in enumerate(temp_a) if temp_b[i] == 1] / mat['Tuning']['SamplingRate'][0][0]

        elif i == 3:

            # VB-Barrel
            mat = io.loadmat('/Volumes/GoogleDrive/My Drive/stp in vivo/data/data_remote/Mar23c1,2,4_Herc_spikes.mat')
            pre = mat['Tlist'][0][0] + .0001 * np.random.random_sample(mat['Tlist'][0][0].shape)
            post = mat['Tlist'][0][1] + .0001 * np.random.random_sample(mat['Tlist'][0][1].shape)
            t_start = np.floor(np.min(np.append(pre, post)))
            pre = pre - t_start
            post = post - t_start

        return pre, post

    #%% main code
if __name__ == "__main__":
    pass
