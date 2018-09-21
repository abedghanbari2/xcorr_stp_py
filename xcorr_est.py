import numpy as np
from scipy import io
import scipy.signal as sps
#import scipy.interpolate as spi
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import time
from patsy import dmatrix
import bisect

class spike_analysis:

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

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def syn(self,xx):
        t_delay = self.tb * self.sigmoid(xx[0])
        tau = .01 * self.sigmoid(xx[1])
        return lambda t : xx[2]*max(0,t-t_delay)/tau * np.exp(1-max(0,t-t_delay)/tau)

    def func_min_xcorr(self,x,bas,y):
        alpha = self.syn(x[-3:])
        synapse = [alpha(i) for i in np.linspace(self.ta,self.tb,self.nbins)]
        lam = np.exp(x[:-3].dot(bas)+synapse)
        return np.sum(-np.multiply(x[:-3].dot(bas)+synapse, y))+np.sum(lam)

    def func_min_glm(self,):
        lam = np.exp()

    def isi_tlist(self,spk):
        isi_pre = np.diff(spk.ravel())
        isi_pre = np.append(np.max(isi_pre),isi_pre)
        return isi_pre

    def spike_transmission_prob(self, min_syn=.0005 , max_syn=.0035 , num_isilog=50 ):
        # calculates efficacy

        isi_pre = self.isi_tlist(self.st1)

        # a log-spaced vector [lowest ISI, highest ISI] presynaptic ISI
        logisi_vec = np.logspace(np.log10(np.min(isi_pre)),np.log10(np.max(isi_pre)),base=10.0,num=num_isilog)

#         for each presynaptic spike what was the ISI before and build a post_spk_presence (0/1)
#         in syn_interval after that presynaptic spike

        post_spk_presence = np.array([1 if np.any( (self.st2>i+min_syn) & (self.st2<i+max_syn)) else 0 for i in self.st1 ])

        isi_split_num = np.digitize(isi_pre,logisi_vec)
        spk_prob = np.zeros(len(logisi_vec))
        t_split_isi = np.zeros(len(logisi_vec))
        for j in range(len(logisi_vec)):
            spk_prob[j] = np.mean(post_spk_presence[isi_split_num==j])
            t_split_isi[j] = np.mean(isi_pre[isi_split_num==j])

        return spk_prob, t_split_isi


    #%% main code
if __name__ == "__main__":

    # load spike times


    # LGN - CTX
#    mat = io.loadmat('/Volumes/GoogleDrive/My Drive/stp in vivo/data/swadlow/July13A.mat')
#    pre = mat['LGN']
#    post = mat['CTX']

     # AVCN
#    mat = io.loadmat('/Volumes/GoogleDrive/My Drive/stp in vivo/data/avcn/Tuning_G1003C82R14.mat')
#    pre = mat['Tuning']['Raw'][0][0]['Continuous'][0][0]['STsEPSP'][0][0]/mat['Tuning']['SamplingRate'][0][0]
#    temp_a = mat['Tuning']['Raw'][0][0]['Continuous'][0][0]['STs'][0][0]
#    temp_b = mat['Tuning']['Raw'][0][0]['Continuous'][0][0]['isAP'][0][0]
#    post = [val for  i,val in enumerate(temp_a) if temp_b[i]==1]/mat['Tuning']['SamplingRate'][0][0]

    # VB-Barrel
    mat = io.loadmat('/Volumes/GoogleDrive/My Drive/stp in vivo/data/data_remote/Mar23c1,2,4_Herc_spikes.mat')
    pre = mat['Tlist'][0][0] + .0001*np.random.random_sample(mat['Tlist'][0][0].shape)
    post = mat['Tlist'][0][1] + .0001*np.random.random_sample(mat['Tlist'][0][1].shape)
    t_start = np.floor(np.min(np.append(pre,post)))
    pre = pre - t_start
    post = post - t_start

    #%% xcorr calculate and estimate synapse
    # cross-correlogram interval and bins
    nbins=200
    ta=-0.005
    tb=.005

    # hyper-parameters of the model
    hyper_rnd_param = 1
    hyper_p = 5 # number of basis (actual bases # = hyper_p+1)
    hyper_rep_opt = 100
    hyper_syn_thr = .1 # threshold for the tranmission interval of the synapse
    hyper_len_fr_bas = 50 # sec
    hyper_cpl_interval = .005 # sec

    # get the histogram
    xcorr_class = spike_analysis(pre,post,ta,tb,nbins)
    xcorr1 = np.histogram(xcorr_class.cross_correlogram(),nbins)

    xx = np.linspace(1,hyper_p-1,nbins)
    bas=np.zeros(shape=(hyper_p+1,nbins))

    for i in range(hyper_p+1):
        bas[i] = sps.bspline(xx-i,3).flatten()
#        plt.plot(bas[i])
#    plt.show()
    # warm-up
    func_val = float('inf')
    t0 = time.time()
    for i in range(hyper_rep_opt):
        x0=np.random.randn(1,np.shape(bas)[0]+3)*hyper_rnd_param
        res_temp = minimize(xcorr_class.func_min_xcorr, x0, method="l-bfgs-b" , args=(bas, xcorr1[0]),
                            options={'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05,
                                     'eps': 1e-08, 'maxfun': 150, 'maxiter': 150, 'iprint': -1,
                                     'maxls': 20})
        if res_temp.fun < func_val:
            func_val = res_temp.fun
            res_warmup = res_temp
    t1 = time.time()
    print('avg time for each loop = ', (t1-t0)/hyper_rep_opt )

    res = minimize(xcorr_class.func_min_xcorr, res_warmup.x, method="l-bfgs-b" , args=(bas, xcorr1[0]),
                            options={'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05,
                                     'eps': 1e-08, 'maxfun': 15000, 'maxiter': 15000, 'iprint': -1,
                                     'maxls': 20})

    # plot xcorr and fits
    t_xcorr = np.linspace(ta,tb,nbins)
    plt.bar(t_xcorr, xcorr1[0], width=t_xcorr[1]-t_xcorr[0])
    plt.plot(t_xcorr, np.exp(res.x[:-3].dot(bas)),color='r',linewidth=3) # the smooth CommInp
    alpha = xcorr_class.syn(res.x[-3:])
    synapse = [alpha(i) for i in t_xcorr] # the synapse
    plt.plot(t_xcorr, np.exp(res.x[:-3].dot(bas)+synapse),color='k',linewidth=3)
    plt.show()

    # plot ISI distribution
    isi_pre = np.diff(pre.ravel())
    isi_pre = np.append(np.max(isi_pre),isi_pre)

    plt.hist(np.log(isi_pre),bins=100)
    plt.show()

    #%% calculate efficacy
    # a log-spaced vector [lowest ISI, highest ISI] presynaptic ISI
    t_syn_interval = t_xcorr[synapse>hyper_syn_thr*np.max(synapse)]
    min_syn = np.min(t_syn_interval)
    max_syn = np.max(t_syn_interval)

    spk_prob, t_split_isi = xcorr_class.spike_transmission_prob( min_syn , max_syn , num_isilog=50 )

    fig, ax = plt.subplots()
    ax.set_xlim([0.0001, 10])
    ax.set_xscale('log')
    for axis in [ax.xaxis]:
        formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
        axis.set_major_formatter(formatter)

    plt.scatter(t_split_isi,spk_prob)
    plt.title("Spike Transmission Probability")
    plt.xlabel("Presynaptic ISI")
    plt.ylabel("Postsynaptic Spiking Prob.")
    plt.show()

    #%% no plasticity GLM

    # get  time_post_before_pre  - time_pre
    max_post_isi = np.max(np.diff(post.ravel()))
    t1=time.time()
    t_post_pre = [post[bisect.bisect_left(post,i)-1]-i for i in pre]
    print('time to calculate t_post_pre : ~',np.ceil(time.time()-t1), ' secs')

    plt.hist(np.log10(np.negative(t_post_pre)),bins=100)
    plt.show()

    #%%    calculate w from synapse interpoation
#    pre = pre[:20]
#    post = post[:20]
    t_post_cpl = [post[bisect.bisect_left(post,i)]-i for i in pre \
        if bisect.bisect_left(post,i)<len(post)]
    t_post_cpl = [i if i<hyper_cpl_interval else None for i in t_post_cpl]
    w = [alpha(i) if i is not None else None for i in t_post_cpl ]

    #%%

#    x = np.power(10,np.linspace(np.log10(.0005),np.log10(.010),100))

    # history covariates
    x = np.negative(t_post_pre)
    knots = np.power(10,np.linspace(np.log10(min(x)+.0001),np.log10(max(x)-.0001),8))
    y = dmatrix("bs(x, degree=3, knots = knots, include_intercept=False) - 1", {"x": x})

    # baseline firing rate covariates
    knots_fr = np.linspace(np.min(pre)+1 , np.max(pre) , \
                           int((np.floor(max(pre)))/hyper_len_fr_bas))
    y_fr = dmatrix("bs(x, degree=3, knots = knots_fr, include_intercept=False) - 1", \
                   {"x": pre})


    Xvar = np.hstack((np.ones([len(w),1]),np.array(y),np.array(y_fr),np.array([w]).T))

    Xvar_notNone = Xvar[[i for i in range(len(Xvar[:,-1])) if Xvar[:,-1][i] is not None],:]

    
