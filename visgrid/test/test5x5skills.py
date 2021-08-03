%matplotlib inline
import random
from visgrid.taxi.taxi import *
from visgrid.taxi.skills import *

random.seed(0)
w = BusyTaxi5x5()
w.plot()

#%%
run_skill(w, 'yellow')
w.plot()

#%%
run_skill(w, 'interact')
w.plot()

#%%
run_skill(w, 'green')
w.plot()

#%%
run_skill(w, 'interact')
w.plot()

#%%
run_skill(w, 'red')
w.plot()

#%%
run_skill(w, 'interact')
w.plot()

#%%
run_skill(w, 'yellow')
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
run_skill(w, 'red')
w.plot()

#%%
run_skill(w, 'interact')
w.plot()

#%%
assert w.check_goal(w.get_state())
