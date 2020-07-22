#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 10:28:52 2020

@author: vipulp
"""
from newSIRD import SIRD_pool, produce_amount_keys
from itertools import product
from multiprocessing import Pool
import pandas as pd
import numpy as np
import bz2
import pickle as pkl

# control number of simulations per job 
block_size = 31
block_size = 1

index = int(sys.argv[1])

def my_product(inp):
    return [dict(zip(inp.keys(), values)) for values in product(*inp.values())]

param_lists = {'inf_rate':[1/8.,1/10.,1/12.,1/14.,1/16.,1/18.,1/20.],\
               'rec_rate':[1/10.,1/13.,1/16.,1/19.,1/22.,1/25.,1/28.,1/31.],\
               'death_rate':[1/500.,1/750.,1/1000.,1/1250.,1/1500.,1/1750.,1/2000.],\
               'scale_pop_dens':[2.,2.5,3.,3.5,4.,4.5,5.,5.5],\
               'scale_med_age':[0.5,0.65,0.8,0.95,1.1,1.25],\
               'age_increase':[5.4,5.5,5.6],\
               'N':[1e5,9e4],\
               'N_init':[1],\
               'sus_age':[0.],\
               'max_age':[100.],\
               'n_sims':[40],\
               'tmax':[100],\
               'tplus':[15],\
               'time_to_fit':[15]}

jobid = os.getenv('SLURM_ARRAY_JOB_ID')
model_list = my_product(param_lists)
# model_names = list(produce_amount_keys(len(model_list)))
random_seeds = range(len(model_list))

# blcks of * sets
start = block_size*(index-1)
end = block_size*index

last = len(model_list)
if end > last:
    end = last
    
model_list = model_list[start:end]
random_seeds = random_seeds[start:end]

for random_seed, model in zip(random_seeds, model_list):
    filename = str(jobid) + "_" + str(index) + "_" + str(i)
    inputs = filename, random_seed, model
    SIRD_pool(inputs)

# inputs = zip(model_names,random_seeds,model_list)

# with Pool(7) as p:
#     print(p.map(SIRD_pool,inputs))

# import glob
# mylist = [f for f in glob.glob("sird_*.bz2")]

# model_results = []
# for filename in mylist:
#     model = {}
#     model_data = pd.read_pickle(filename,compression='infer')
#     parm_list = list(model_data.columns)
#     for par in param_lists.keys():
#         model[par] = model_data[par].unique()[0]
#     model_slopes,model_intercepts,model_t0 = model_data[['slope']].to_numpy(),model_data[['intercept']].to_numpy(),\
#         model_data[['t0']].to_numpy()
#     model['id'],model['log_slope'],model['log_slope_sigma'],model['intercept'],model['intercept_sigma'],\
#         model['t0'],model['t0_sigma'] = \
#         filename[:-4],np.log(model_slopes).mean(),np.log(model_slopes).std(),model_intercepts.mean(),model_intercepts.std(),\
#             model_t0.mean(),model_t0.std()
#     model_results.append(dict(model))
# sim_df = pd.DataFrame(model_results)
# with bz2.BZ2File('sim_summary.bz2', 'w') as pik:
#     pkl.dump(sim_df,pik)