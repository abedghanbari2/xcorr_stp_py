from xcorr_est import SpikeAnalysis
import numpy as np
import matplotlib.pyplot as plt

#%% xcorr calculate and estimate synapse
# cross-correlogram interval and bins
nbins=200
ta=-0.005
tb=0.005

# get the histogram
pre, post = SpikeAnalysis.load_data(1)
pre=pre[:8000,0]
post = post[post<np.max(pre)]
xcorr_class = SpikeAnalysis(pre,post,ta,tb,nbins)
t_pre_cpl = xcorr_class.time_pre_cpl()
Y = xcorr_class.y_mat(t_pre_cpl, num_bins=100)


res, synapse, alpha, slow_xcorr, xcorr, t_xcorr = xcorr_class.estimate_synapse(num_basis=5, num_rept=100, rnd_scale=1, plot_flag=1)

X = xcorr_class.make_X(res.x[-3:], max_history_filter=.01, num_history_splines=20, len_fr_bas=200,plot_h_flag=1)

res = xcorr_class.optim_func(X, synapse, Y, 4)

print(xcorr_class.transf_theta(res.x[-6:-1]))

# plot the estimated efficacy curves
mean_eff, isi_vec = xcorr_class.spike_trans_prob_est(res.x, X, synapse, N=40, plot_flag=1)