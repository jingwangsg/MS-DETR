#%% 
from kn_util.basic import load_pickle
ret_list = load_pickle("../_debug/ret_list.pkl")
# %%
import numpy as np
np.array([_["ious"][0] for _ in ret_list])
# %%
