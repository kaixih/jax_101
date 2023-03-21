from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
import jax
import jax.numpy as jnp

def show_ir(f, *args):
  lowered = f.lower(*args)
  print(lowered.as_text())
  compiled = lowered.compile()
  print(compiled.as_text())

@jax.jit
def jitted_cast(x, scale):
  casted_x = x * scale
  amax = jnp.max(jnp.abs(x * scale))
  return casted_x, amax

@jax.jit
def jitted_dot(casted_x, scale_x, casted_w, scale_w):
  scale_inv_x = 1.0 / scale_x
  scale_inv_w = 1.0 / scale_w
  y = jnp.dot(casted_x * scale_inv_x, casted_w * scale_inv_w)
  return y

sharding = PositionalSharding(mesh_utils.create_device_mesh((8,)))
sharding = sharding.reshape(4, 2)

x = jax.random.normal(jax.random.PRNGKey(0), (8192, 784))
x = jax.device_put(x, sharding.replicate(1))

scale_x = 1.0
scale_w = 1.0

def cast_and_dot(x, scale_x, w, scale_w, debug_on):
  if debug_on:
    show_ir(jitted_cast, x, scale_x)
    show_ir(jitted_cast, w, scale_w)
  casted_x, amax_x = jitted_cast(x, scale_x)
  casted_w, amax_w = jitted_cast(w, scale_w)
  # ... Update the scale using amax
  if debug_on:
    show_ir(jitted_dot, casted_x, scale_x, casted_w, amax_w)
    jax.debug.visualize_array_sharding(casted_x)
    jax.debug.visualize_array_sharding(casted_w)
  y = jitted_dot(casted_x, scale_x, casted_w, amax_w)
  if debug_on:
    jax.debug.visualize_array_sharding(y)
  return y

w1 = jax.random.normal(jax.random.PRNGKey(0), (784, 8192))
w1 = jax.device_put(w1, sharding.replicate())
y = cast_and_dot(x, scale_x, w1, scale_w, debug_on=False)

w2 = jax.random.normal(jax.random.PRNGKey(0), (8192, 8192))
w2 = jax.device_put(w2, sharding.replicate(0))
y = cast_and_dot(y, scale_x, w2, scale_w, debug_on=False)

w3 = jax.random.normal(jax.random.PRNGKey(0), (8192, 8192))
w3 = jax.device_put(w3, sharding.replicate(0).T)
y = cast_and_dot(y, scale_x, w3, scale_w, debug_on=False)

w4 = jax.random.normal(jax.random.PRNGKey(0), (8192, 10))
w4 = jax.device_put(w4, sharding.replicate())
y = cast_and_dot(y, scale_x, w4, scale_w, debug_on=True)

