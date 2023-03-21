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

x = jax.random.normal(jax.random.PRNGKey(0), (8192, 784))
x = jax.device_put(x, sharding.replicate(1))

def dot_wrapper(x, w, debug_on):
  if debug_on:
    show_ir(jitted_dot, x, w)
    jax.debug.visualize_array_sharding(x)
    jax.debug.visualize_array_sharding(w)
  y = jitted_dot(x, w)
  if debug_on:
    jax.debug.visualize_array_sharding(y)
  return y

w1 = jax.random.normal(jax.random.PRNGKey(0), (784, 8192))
w1 = jax.device_put(w1, sharding.replicate())
y = dot_wrapper(x, w1, debug_on=False)

w2 = jax.random.normal(jax.random.PRNGKey(0), (8192, 8192))
w2 = jax.device_put(w2, sharding.replicate(0))
y = dot_wrapper(y, w2, debug_on=False)

w3 = jax.random.normal(jax.random.PRNGKey(0), (8192, 8192))
w3 = jax.device_put(w3, sharding.replicate(0).T)
y = dot_wrapper(y, w3, debug_on=False)

w4 = jax.random.normal(jax.random.PRNGKey(0), (8192, 10))
w4 = jax.device_put(w4, sharding.replicate())
y = dot_wrapper(y, w4, debug_on=True)

