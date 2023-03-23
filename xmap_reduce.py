import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.maps import xmap
from jax.sharding import Mesh

def show_ir(f, x):
  lowered = f.lower(x)
  print(lowered.as_text())
  compiled = lowered.compile()
  print(compiled.as_text())

def mixed_loss(x):
  y = jnp.sum(x, axis=0)
  return jax.lax.psum(y, 'inputs')

def named_loss(x):
  return jax.lax.psum(x, 'inputs')

def default_loss(x):
  return jnp.sum(x)

in_axes = [['batch', 'inputs', ...]]
out_axes = ['batch', 'inputs', ...]

mixed_l = xmap(mixed_loss, in_axes=in_axes, out_axes=out_axes,
               axis_resources={'inputs': 'x'})
named_l = xmap(named_loss, in_axes=in_axes, out_axes=out_axes,
               axis_resources={'inputs': 'x'})
default_l = xmap(default_loss, in_axes=in_axes, out_axes=out_axes,
                 axis_resources={'inputs': 'x'})

x = jax.random.normal(jax.random.PRNGKey(0), (3, 16))

devices = np.array(jax.local_devices())
with Mesh(devices, ('x',)):
  x = x.reshape((3, 8, 2))
  # For each row, each gpu reduces its sharded two values and then reduce across
  # gpus to a single value. Then the single value is broadcast to 8.
  y0 = mixed_l(x)
  print("mixed shape:", y0.shape)
  print("mixed values:", y0)

  x = x.reshape((3, 16))
  # For each row, all gpus reduces the 16 values to a single value. Then the
  # single value is broadcast to 16.
  y0 = named_l(x)
  print("named shape:", y0.shape)
  print("named values:", y0)

  x = x.reshape((3, 8, 2))
  # For each row, each gpu reduces its sharded two values. There is no reduce
  # across gpus. Each gpu stores the values and so we get 8 different values.
  y0 = default_l(x)
  print("named shape:", y0.shape)
  print("named values:", y0)
