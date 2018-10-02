from xcorr_est import SpikeAnalysis
import numpy as np
import matplotlib.pyplot as plt

#%% xcorr calculate and estimate synapse
# cross-correlogram interval and bins
nbins=200
ta=-0.005
tb=0.005

# get the histogram
pre, post = SpikeAnalysis.load_data(3)
pre=pre[:8000,0]
post = post[post<np.max(pre)]
xcorr_class = SpikeAnalysis(pre,post,ta,tb,nbins)
t_pre_cpl = xcorr_class.time_pre_cpl()
Y = xcorr_class.y_mat(t_pre_cpl, num_bins=100)
res, synapse, alpha, slow_xcorr, xcorr, t_xcorr = xcorr_class.estimate_synapse(num_basis=5, num_rept=100, rnd_scale=1, plot_flag=1)
X = xcorr_class.make_X(res.x[-3:], max_history_filter=.01, num_history_splines=20, len_fr_bas=50, plot_h_flag=1)

#%%

res = xcorr_class.optim_func(X, synapse, Y, 4)

# plot the estimated efficacy curves
mean_eff, isi_vec = xcorr_class.spike_trans_prob_est(res.x, X, synapse, N=100, plot_flag=1)

t_syn_interval = t_xcorr[synapse>.1*np.max(synapse)]
spk_prob, t_split_isi = xcorr_class.spike_transmission_prob(np.min(t_syn_interval), np.max(t_syn_interval), num_isilog=50, plot_flag=1)

#%% GLM
from glm.glm import GLM
from glm.families import Bernoulli

bern_model = GLM(family=Bernoulli())
bern_model.fit(X, np.sum(Y,axis=1))
bern_model.coef_

x_glm = np.random.randn(np.shape(X)[1] + 6)
x_glm[:-6] = bern_model.coef_
# plot the estimated efficacy curves
mean_eff, isi_vec = xcorr_class.spike_trans_prob_est(x_glm, X, synapse, N=100, plot_flag=1)

t_syn_interval = t_xcorr[synapse>.1*np.max(synapse)]
spk_prob, t_split_isi = xcorr_class.spike_transmission_prob(np.min(t_syn_interval), np.max(t_syn_interval), num_isilog=50, plot_flag=1)
