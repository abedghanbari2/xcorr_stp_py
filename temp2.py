#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 19:25:39 2018

@author: abedghanbari
"""

    @staticmethod
    def cost_function(b,X,y):
        lam = np.exp(np.dot(X,b))
        return -np.dot(np.log(lam),y)+np.sum(lam)
    @staticmethod
    def cost_function_derv(b,X,y):
        lam = np.exp(np.dot(X,b))
        return np.dot((lam-y),X) 

    # load spike times


    # LGN - CTX
    mat = io.loadmat('/Volumes/GoogleDrive/My Drive/stp in vivo/data/swadlow/July13A.mat')
    pre = mat['LGN'][:-1]
    post = mat['CTX']

     # AVCN
#    mat = io.loadmat('/Volumes/GoogleDrive/My Drive/stp in vivo/data/avcn/Tuning_G1003C82R14.mat')
#    pre = mat['Tuning']['Raw'][0][0]['Continuous'][0][0]['STsEPSP'][0][0]/mat['Tuning']['SamplingRate'][0][0]
#    temp_a = mat['Tuning']['Raw'][0][0]['Continuous'][0][0]['STs'][0][0]
#    temp_b = mat['Tuning']['Raw'][0][0]['Continuous'][0][0]['isAP'][0][0]
#    post = [val for  i,val in enumerate(temp_a) if temp_b[i]==1]/mat['Tuning']['SamplingRate'][0][0]

    # VB-Barrel
#    mat = io.loadmat('/Volumes/GoogleDrive/My Drive/stp in vivo/data/data_remote/Mar23c1,2,4_Herc_spikes.mat')
#    pre = mat['Tlist'][0][0] + .0001*np.random.random_sample(mat['Tlist'][0][0].shape)
#    post = mat['Tlist'][0][1] + .0001*np.random.random_sample(mat['Tlist'][0][1].shape)
#    t_start = np.floor(np.min(np.append(pre,post)))
#    pre = pre - t_start
#    post = post - t_start

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
    hyper_len_fr_bas = 200 # sec
    hyper_cpl_interval = .005 # sec
    hyper_rnd_param_glm = 1
    hyper_max_history_filter = .1
    hyper_num_history_splines = 20
    
    # get the histogram
    xcorr_class = SpikeAnalysis(pre,post,ta,tb,nbins)
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

    # %% calculate efficacy
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

    # %%
    # no plasticity GLM

    # get  time_post_before_pre  - time_pre
    max_post_isi = np.max(np.diff(post.ravel()))
    t1=time.time()
    t_post_pre = [post[bisect.bisect_left(post,i)-1]-i for i in pre]
    print('time to calculate t_post_pre : ~',np.ceil(time.time()-t1), ' secs')

#    plt.hist(np.log10(np.negative(t_post_pre)),bins=100)
#    plt.show()
    #    calculate w from synapse interpoation
    t_post_cpl = [post[bisect.bisect_left(post,i)]-i for i in pre  if bisect.bisect_left(post,i)<len(post)]
    y = [1 if ((i>min_syn) & (i<max_syn)) else 0 for i in t_post_cpl]
    # len(y)
    t_post_cpl = [i if i<hyper_cpl_interval else None for i in t_post_cpl]
    w = [alpha(i).item() if i is not None else None for i in t_post_cpl ]
    
    # history covariates
    x = np.negative(t_post_pre)
    x[x<0]=np.median(x[x>0])
#    knots = np.power(np.exp(1),np.linspace(np.log10(min(x)+.0001),np.log10(hyper_max_history_filter-.0001),8))
#%%
    knots_history = np.linspace(min(x)+.0001,hyper_max_history_filter -.0001,hyper_num_history_splines)
    X_h = dmatrix("bs(x, degree=3, knots = knots_history, include_intercept=False) - 1", {"x": x})
    for i in range(X_h.shape[1]-4):
        plt.scatter(x,X_h[:,i]) 
    plt.xlabel('pre spike time - previous postsynaptic spike')
    plt.xlim([0,.1])
    plt.show()
    X_h_truncated = X_h[:,:hyper_num_history_splines-4]
#%%
    # baseline firing rate covariates
    knots_fr = np.linspace(np.min(pre)+1, np.max(pre), int((np.floor(max(pre)))/hyper_len_fr_bas))
    X_fr = dmatrix("bs(x, degree=3, knots = knots_fr, include_intercept=False) - 1", {"x": pre})
#    Xvar = np.hstack((np.ones([len(w),1]),np.array(X_h),np.array(X_fr),np.array([w]).T))
    X_fr_truncated = X_fr[:,:-1]
    Xvar = np.hstack((np.ones([len(w),1]),np.array(X_fr_truncated),np.array(X_h_truncated),np.array([w]).T))
#    Xvar = np.hstack((np.ones([len(w),1]),np.array([w]).T))
    X_notNone = Xvar[[i for i in range(len(Xvar[:,-1])) if Xvar[:,-1][i] is not None],:].astype('float')
    y_notNone = np.array([y[i] for i in range(len(y)) if w[i] is not None])
#    X_notNone = X_notNone[:,:-1]
    pre_notNone = np.array([pre[i] for i in range(len(pre)) if w[i] is not None])
 #%% generalized linear model 
    
    from glm.glm import GLM
    from glm.families import Poisson
    poisson_model = GLM(family=Poisson())
#    poisson_model.fit(Xvar.astype('float'), np.array(y))
    t0 = time.time()
    poisson_model.fit(X_notNone, y_notNone)
    print('time to finish optimization: '+str(time.time()-t0)+' sec')
    poisson_model.coef_
    
    x_rate = np.linspace(np.min(pre)+1, np.max(pre), 100)
    X_pred = dmatrix("bs(x, degree=3, knots = knots_fr, include_intercept=False) - 1", {"x": x_rate})
    
    plt.plot(x_rate[:-1],np.exp(poisson_model.coef_[0]+np.dot(X_pred[:-1,:-1],poisson_model.coef_[1:X_fr_truncated.shape[1]+1])))
    plt.xlabel('Time [sec]')
    plt.ylabel('Rate (Hz)')    

#    %%
    b0 = np.zeros(X_notNone.shape[1])
    t0 = time.time()
    es_temp = minimize(xcorr_class.cost_function, b0, args=(X_notNone, y_notNone), method="l-bfgs-b", jac=xcorr_class.cost_function_derv)
    print('time to finish optimization: '+str(time.time()-t0)+' sec')
    plt.plot(x_rate[:-1],np.exp(es_temp.x[0]+np.dot(X_pred[:-1,:-1],es_temp.x[1:X_fr_truncated.shape[1]+1])))
    plt.xlabel('Time [sec]')
    plt.ylabel('Rate (Hz)')
    plt.legend(['PyGLM','Ours'])
    plt.title('Slow changes in firing rate of presynaptic spike')
    plt.show()
    
#    x_history = np.linspace(min(knots_history)-.0001, max(knots_history)+.0001, 100)
#    X_history_plot = dmatrix("bs(x, degree=3, knots = knots_history, include_intercept=False) - 1", {"x": x_history})
    
#    plt.plot(x_history,X_history_plot[:,:-4])

#    plt.plot(x_history[:-1],np.exp(np.dot(X_history_plot[:-1,:-7],es_temp.x[X_fr_truncated.shape[1]+1:-1])))
#    plt.plot(x_history[:-1],np.exp(np.dot(X_history_plot[:-1,:-7],poisson_model.coef_[X_fr_truncated.shape[1]+1:-1])))
#    plt.legend(['GLM','Ours'])
#    plt.ylim([0,2])
#    plt.show()

#%%
    bin_length = 10
    plt.plot(x_rate[:-1],np.exp(es_temp.x[0]+np.dot(X_pred[:-1,:-1],es_temp.x[1:X_fr_truncated.shape[1]+1])))
    bin_vec = np.linspace(0,np.ceil(np.max(pre)),int(np.ceil(np.max(pre)/bin_length)))
    hist_fr = np.histogram(post,bins=bin_vec)
    plt.plot(bin_vec[:-1],hist_fr[0]/bin_length)
    plt.show()a