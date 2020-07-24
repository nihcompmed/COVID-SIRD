#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 10:58:28 2020

@author: vipulp
"""
import math
import numpy as np
import scipy.optimize as spopt
import scipy.stats as spstats
import networkx as nx
from collections import defaultdict,Counter
import EoN
import string
import secrets
import pandas as pd
import pickle as pkl
import bz2

max_age_=100.
ages = np.arange(1.,max_age_+1.,1.)

def b36(n, N, chars=string.ascii_uppercase + string.digits):
    s = ''
    for _ in range(N):
        s += chars[n % 36]
        n //= 36
    return s

def produce_amount_keys(amount_of_keys): #this generates random strings for labeling the results of the models
    keys = set()
    while len(keys) < amount_of_keys:
        N = np.random.randint(12, 20)
        keys.add(b36(secrets.randbelow(36**N), N))
    return keys

def changenum(data):
    return int(data.replace(',',''))
def elements(array): #check if array exists
    return array.ndim and array.size


def sigmoid(ro):
    # x is expected to be ages/max_age
    r1 = np.abs(ro)
    c1 = r1[0]
    beta = r1[1]
    def f(x):
        p = 1/(1.+c1*x**beta)
        p -= p[-1]
        return p/np.sum(p)
    return f

def population_density_func(dict_p): #sma = scale_med_age,coefficients from fitting world demographic data
    sma = dict_p['scale_med_age']
    max_age = dict_p['max_age']
    return sigmoid([5*max_age/(3.*40*sma),10/3.])(ages/max_age)


def death_weight(dict_p): #susceptibility, age_exp
    max_age = dict_p['max_age']
    ro = np.zeros(2)
    med_age_ny = 38.7
    ro[0] = 5/3.*max_age/med_age_ny
    ro[1] = 10/3.
    age_probs_ny = sigmoid(ro)(ages/max_age)
    sus = dict_p['sus_age']
    age_exp = dict_p['age_increase']
    death_probs = (ages/max_age)**age_exp
    norm = np.sum(age_probs_ny*death_probs)
    death_probs /= norm
    # ages_sus = ages<(sus)
    # death_probs = np.ones(ages.shape,dtype=np.float)
    # death_probs[~ages_sus] = ((ages[~ages_sus]/(sus))**age_exp)
    # death_probs /= 4190. #from NY state median_age = 38.7 normalizing death_probs*pop_dens
    return death_probs

def generate_graph(dict_p,graph_seed=-1): # N = # of nodes, spd = links per node in a G(N,p) graph
  N = int(dict_p['N'])
  spd = dict_p['scale_pop_dens']
  if graph_seed>=0:
      return nx.fast_gnp_random_graph(N, 2*(spd/(N-1)),seed=graph_seed)
  else:
      return nx.fast_gnp_random_graph(N, 2*(spd/(N-1)))

def set_node_status(graph,dict_p,node_seed=-1):
  max_age = dict_p['max_age']
  node_age_probs = population_density_func(dict_p)
  death_probs = death_weight(dict_p)
  N = graph.number_of_nodes()
  if node_seed >= 0:
      np.random.seed(node_seed)
  node_age_choice = np.random.choice(np.arange(1,int(max_age)+1,dtype=np.int),size=N,p=node_age_probs)
  node_ages = {node: node_age_choice[i] for i,node in enumerate(graph.nodes())}
  # node_attribute_dict = {node: max(1.,(node_ages[node]/(sus))**age_exp) for node in graph.nodes()}
  node_attribute_dict = {node: death_probs[int(node_ages[node])-1] for node in graph.nodes()}
  nx.set_node_attributes(graph, values=node_attribute_dict, name='expose2infect_weight')


def SIRD(graph, dict_p, tmax = 100, IC = {}, node_seed=-1):
    for key in dict_p.keys():
        if key == 'inf_rate':
            inf_rate = dict_p['inf_rate']
        elif key == 'rec_rate':
            rec_rate = dict_p['rec_rate']
        elif key == 'death_rate':
            death_rate = dict_p['death_rate']
        elif key == 'N_init':
            N_init = dict_p['N_init']
    
    H = nx.DiGraph()  #the spontaneous transitions
    H.add_edge('Inf', 'Dead', rate = death_rate,weight_label='expose2infect_weight')
    H.add_edge('Inf', 'Rec', rate = rec_rate)
   
    J = nx.DiGraph()  #the induced transitions
    J.add_edge(('Inf', 'Sus'), ('Inf', 'Inf'), rate = inf_rate)
    if len(IC) == 0:
        set_node_status(graph, dict_p,node_seed=node_seed)
        IC = defaultdict(lambda:'Sus')
        for i in list(np.random.randint(dict_p['N'],size = math.ceil(N_init))):
            IC[i] = 'Inf'
      
    return_statuses = ['Sus', 'Inf', 'Rec', 'Dead']
    sim = EoN.Gillespie_simple_contagion(graph, H, J, IC, return_statuses, tmax=tmax, \
                                         return_full_data=True)
    return sim,dict_p

def decimate_edges(graph,fract,decimate_seed=-1): #fract = fraction edges to remove
    #first do a graph deep copy, then remove fract fraction of edges
    G = graph.copy()
    if decimate_seed>=0: np.random.seed(decimate_seed)
    G.remove_edges_from([x for x in G.edges if np.random.uniform() < fract])
    return G

def idiot_nodes(graph,fract_idiots,fract_decimate,decimate_seed=-1):
    G = graph.copy()
    if decimate_seed>=0: np.random.seed(decimate_seed)
    good_nodes = [x for x in list(G.nodes()) if np.random.uniform() > fract_idiots]
    G.remove_edges_from([x for x in G.edges() if (((x[0] in good_nodes) or (x[1] in good_nodes)) \
                                                and (np.random.uniform() < fract_decimate))])
    return G

def return_counts(dic,stat):
    return Counter(dic.values())[stat]


def slope_from_sim(sim,dict_p):
    tm, D = sim.summary()
    time_to_fit = dict_p['time_to_fit']
    dm = D['Dead']
    return slope_from_data(tm, dm, time_to_fit)

def slope_from_data(tm,dm,time_to_fit):
    dead_gt_0 = dm > 0
    if elements(tm[dead_gt_0])>0:
        time0 = tm[dead_gt_0][0]
        t_15 = (tm[dead_gt_0]-time0) < time_to_fit
        t15,d15 = (tm[dead_gt_0]-time0)[t_15],dm[dead_gt_0][t_15]
        if elements(t15)>1: #more than one point with dead
            lslope,c_model,r,p,sig_m = spstats.linregress(t15,np.log(d15))
            if lslope > 0:
                m_model,c_model = lslope,c_model  
            else:
                m_model,c_model = 1e-12,0.0
            return m_model,c_model,t15,d15,time0
        else: return 1e-12,0.0,t15,d15,time0
    else: return np.nan,np.nan,np.nan,np.nan,np.nan #no deaths

def statuses_from_sim(sim,time_to_plot=50.):  #this can be used to plot 
    status_50 = [sim.get_statuses(time = i) for i in range(int(time_to_plot))]
    return status_50

def continue_sim(graph,dict_p,sim,mult): #starting the simulation continuation at time of first death + mult X time_to_fit
    IC = sim.get_statuses(time=dict_p['t0']+mult*dict_p['time_to_fit'])
    return SIRD(graph,dict_p,tmax=dict_p['tplus'],IC=IC)

def SIRD_pool(inp):
    name_id, random_seed, model0 = inp
    pickle_name = 'pickles/sird_' + name_id + '.bz2'
    np.random.seed(random_seed)
    model_results = []
    for i in range(model0['n_sims']):
        model = dict(model0)
        graph_init = generate_graph(model)
        sim_init,model = SIRD(graph_init,model)
        model['slope'],model['intercept'],model['times15'],model['dead15'],model['t0'] = slope_from_sim(sim_init,model)
        model['time_courses'] = [dict(Counter(sim_init.get_statuses(time=i).values())) for i in range(model['tmax'])]
        if not np.isnan(model['slope']):
            #for mult in []:#[1,2,3]:
            for mult in [1,2,3]:
                for frac_idiot,frac_del in [(1/3,1/2),(1/3,2/3),(1/4,1/2),(1/4,2/3)]:
                    if frac_idiot ==1/3: 
                        str_idiot = 'third'
                    else:
                        str_idiot = 'quarter'
                    if frac_del == 1/2: 
                        str_del = 'half'
                    else:
                        str_del = 'twothird'
                    idiot_str = 'idiot'+str(mult)+'_'+ str_idiot +'_'+ str_del
                    graph_idiot = idiot_nodes(graph_init,frac_idiot,frac_del)
                    sim_idiot,model = continue_sim(graph_idiot,model,sim_init,mult)
                    model[idiot_str] = [dict(Counter(sim_idiot.get_statuses(time=i).values())) for i in range(model['tplus'])]
            model_results.append(model)
    if len(model_results)>1:
        df = pd.DataFrame(model_results)
        with bz2.BZ2File(pickle_name, 'w') as pik:
            pkl.dump(df,pik)
        return name_id
    return ' '
# dict_try = {'N':1e2,'scale_pop_dens':3,'sus_age':10.,'age_increase':5,'scale_med_age':1.0}
# graph = generate_graph(dict_try)
# sim,dict_try = SIRD(graph,dict_try)
# sim,dict_try = SIRD(decimate_edges(graph,0.5,decimate_seed=1),dict_try)
# print(slope_from_sim(sim))
# print(statuses_from_sim(sim,time_to_plot=5))
# sim,dict_try = continue_sim(graph,dict_try,sim,15,25)
# print(slope_from_sim(sim))

