%matplotlib inline
import random
from visgrid.taxi.skills import *
from visgrid.taxi.taxi import *

random.seed(0)
w = Taxi10x10()
w.plot()

#%%
run_skill(w, 'gray')
w.plot()

#%%
run_skill(w, 'interact')
w.plot()

#%%
run_skill(w, 'blue')
w.plot()

#%%
run_skill(w, 'interact')
w.plot()

#%%

assert w.check_goal(w.get_state())
