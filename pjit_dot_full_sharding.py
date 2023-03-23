from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
import jax
import jax.numpy as jnp

def show_ir(f, x, w):
  lowered = f.lower(x, w)
  print(lowered.as_text())
  compiled = lowered.compile()
  print(compiled.as_text())

@jax.jit
def jitted_dot(x, w):
  y = jnp.dot(x, w)
  return y

sharding = PositionalSharding(mesh_utils.create_device_mesh((8,)))
sharding = sharding.reshape(4, 2)

def dot_wrapper(x, w, debug_on):
  if debug_on:
    show_ir(jitted_dot, x, w)
    jax.debug.visualize_array_sharding(x)
    jax.debug.visualize_array_sharding(w)
  y = jitted_dot(x, w)
  if debug_on:
    jax.debug.visualize_array_sharding(y)
  return y

x = jax.random.normal(jax.random.PRNGKey(0), (8192, 8192))
x = jax.device_put(x, sharding)
w1 = jax.random.normal(jax.random.PRNGKey(0), (8192, 8192))
w1 = jax.device_put(w1, sharding.T)
y = dot_wrapper(x, w1, debug_on=True)

