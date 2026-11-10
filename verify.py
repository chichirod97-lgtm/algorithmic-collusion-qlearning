"""Economic and algorithmic checks before simulation interpretation."""
import numpy as np
from model import Config,environment,shares,run
c=Config();g,p,r=environment(c)
assert np.allclose(p[:,:,0],p[:,:,1].T)
assert abs(r['nash_price']-1.47293)<1e-4
assert abs(r['monopoly_price']-1.92498)<1e-4
assert len(g)==15 and g[0]<r['nash_price']<r['monopoly_price']<g[-1]
# A firm cannot increase profit by a small deviation at the continuous Nash price.
n=r['nash_price']; own=np.linspace(n-.01,n+.01,101)
pi=(own-c.cost)*shares(own,n,c)[0]
assert np.argmax(pi)==50
a=run(c,123,max_steps=1000);b=run(c,123,max_steps=1000)
for field in ['frozen_mean_price','profit_gain','history','cycle_length']:
    assert a[field]==b[field]
assert not a['converged']
print('PASS: symmetry, economic reference prices, best response, seeded repeatability, and nonconvergence status.')
