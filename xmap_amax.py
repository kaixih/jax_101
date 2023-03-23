import jax.numpy as jnp
import jax
import numpy as np
from typing import Any, Callable
from jax.sharding import Mesh
from jax.experimental.maps import xmap

# This mimics the calculation of amax in a distributed scenario. Here, the local
# variable amax is for the local shard. After the collective op pmax, we can get
# the amax for the logical ndarray.
def foo(x, y):
  amax = jnp.max(y)
  y += 1.0
  return jax.lax.pmax(amax, 'batch'), y

amax_xmap = xmap(foo,
                 in_axes=({}, {0 : 'batch'}),
                 out_axes=({}, {0 : 'batch'}),
                 axis_resources={'batch' : 'x'})

old_amax = jnp.zeros((1,))
x = jax.random.normal(jax.random.PRNGKey(0), (4, 20))

devices = np.array(jax.local_devices()).reshape((4, 2))
with Mesh(devices, ('x', 'y')):
  new_amax, y = amax_xmap(old_amax, x)

print(new_amax.shape)
print("Result from xmap:", new_amax)
print("Result for reference:", jnp.max(x))
print(y.shape)
print(y)
assert (y == (x + 1.0)).all()

