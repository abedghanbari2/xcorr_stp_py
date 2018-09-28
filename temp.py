

class SpikeAnalysis:

    def __init__(self, st1,st2,ta=-.1,tb=.1, nbins=100):
        self.st1 = st1
        self.st2 = st2
        self.ta = ta
        self.tb = tb
        self.nbins = nbins

    def cross_correlogram(self):
        all=[]
        for ref in self.st1:
            all.extend(list(self.st2[(self.st2>ref+self.ta)&(self.st2<ref+self.tb)]-ref))
        return all

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

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
    def func_min_glm(x, y_notNone, Xvar_notNone):
        lam = np.exp(x.dot(Xvar_notNone)[0].ravel())
        return np.sum(-np.multiply(lam, y_notNone)) + np.sum(lam)

    @staticmethod
    def func_min_glm_der(x, y_not_none, Xvar_notNone):
        lam = np.exp(x.dot(Xvar_notNone)[0].ravel())
        for i in range(len(x)):
            derv_x[i] = np.sum(-np.multiply(y_not_none, Xvar_notNone[i, :])) + np.sum(
                np.multiply(lam, Xvar_notNone[i, :])))
            return derv_x

    def isi_tlist(self, spk):
        isi_pre = np.diff(spk.ravel())
        isi_pre = np.append(np.max(isi_pre), isi_pre)
        return isi_pre

    def spike_transmission_prob(self, min_syn=.0005, max_syn=.0035, num_isilog=50):
        # calculates efficacy

        isi_pre = self.isi_tlist(self.st1)

        # a log-spaced vector [lowest ISI, highest ISI] presynaptic ISI
        logisi_vec = np.logspace(np.log10(np.min(isi_pre)), np.log10(np.max(isi_pre)), base=10.0, num=num_isilog)

        #         for each presynaptic spike what was the ISI before and build a post_spk_presence (0/1)
        #         in syn_interval after that presynaptic spike

        post_spk_presence = np.array(
            [1 if np.any((self.st2 > i + min_syn) & (self.st2 < i + max_syn)) else 0 for i in self.st1])

        isi_split_num = np.digitize(isi_pre, logisi_vec)
        spk_prob = np.zeros(len(logisi_vec))
        t_split_isi = np.zeros(len(logisi_vec))
        for j in range(len(logisi_vec)):
            spk_prob[j] = np.mean(post_spk_presence[isi_split_num == j])
            t_split_isi[j] = np.mean(isi_pre[isi_split_num == j])

        return spk_prob, t_split_isi


a = SpikeAnalysis(1,2,3,4,5)
a.chert()