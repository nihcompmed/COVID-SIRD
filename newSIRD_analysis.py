#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:58:53 2020

@author: vipulp
"""
from newSIRD import changenum,slope_from_data
import bz2
import pickle as pkl
import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import Pool
from math import ceil

import glob
# params_to_track = ['inf_rate', 'rec_rate','death_rate', 'scale_pop_dens','scale_med_age','age_increase',\
#                    'sus_age','max_age','n_sims','tmax','tplus','time_to_fit','N']
params_to_track = ['inf_rate', 'rec_rate','death_rate', 'scale_pop_dens','scale_med_age','age_increase',\
                    'sus_age','max_age','n_sims','tmax','tplus','time_to_fit','N']
mylist = [f for f in glob.glob("sird_*.bz2")]

    
def extract_key(key,lst):
    return [x[key] for x in lst]

def SIRD_summary(filename):
    initial_dict = {'Sus':0,'Inf':0,'Dead':0,'Rec':0}
    def uniform_dict(dic):
        unif_dict = dict(initial_dict)
        for key in dic.keys():
            unif_dict[key] = dic[key]
        return unif_dict
    model = {}
    model_data = pd.read_pickle(filename,compression='infer')
    parm_list = list(model_data.columns)
    for par in params_to_track:
        model[par] = model_data[par].unique()[0]
    model_slopes,model_intercepts,model_t0 = model_data[['slope']].to_numpy(),model_data[['intercept']].to_numpy(),\
        model_data[['t0']].to_numpy()
    model['id'],model['log_slope'],model['log_slope_sigma'],model['intercept'],model['intercept_sigma'],\
        model['t0'],model['t0_sigma'] = \
        filename[:-4],np.log(model_slopes).mean(),np.log(model_slopes).std(),model_intercepts.mean(),model_intercepts.std(),\
            model_t0.mean(),model_t0.std()

    for y in ['time_courses', 'idiot_third_half', 'idiot_third_twothird', 'idiot_quarter_half', 'idiot_quarter_twothird',\
              'idiot2_third_half', 'idiot2_third_twothird', 'idiot2_quarter_half', 'idiot2_quarter_twothird',\
              'idiot3_third_half', 'idiot3_third_twothird', 'idiot3_quarter_half', 'idiot3_quarter_twothird']:
        if y in parm_list:
            model[y] = dict(initial_dict)
            list_time_courses = [list(map(uniform_dict,x)) for x in list(model_data[y])]
            for key in initial_dict.keys():
                model[y][key] = np.array([extract_key(key, x) for x in list_time_courses])
    list_of_shifted_times = []
    # print(model['n_sims'],model['time_courses']['Dead'].shape[0])
    for sim in range(model['time_courses']['Dead'].shape[0]): #because not all simulations have deaths
        list_of_shifted_times.append(np.arange(model['tmax'])-model_t0[sim])
    model['shifted_times'] = np.array(list_of_shifted_times)
    return model

with Pool(7) as p:
    sim_df = pd.DataFrame(p.map(SIRD_summary,mylist))
with bz2.BZ2File('sim_summary.bz2', 'w') as pik:
    pkl.dump(sim_df,pik)



