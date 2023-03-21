import jax
import jax.numpy as jnp

global_list = []

def log2(x):
  # Side-effect code will be executed but won't be traced in the JAXPR.
  global_list.append(x)
  ln_x = jnp.log(x)
  ln_2 = jnp.log(2.0)
  return ln_x / ln_2

print(jax.make_jaxpr(log2)(3.0))
print(len(global_list))

def log2_with_print(x):
  # Side-effect code will be executed once during tracing but the input x has
  # already been converted to a tracer object.
  print("printed x:", x)
  ln_x = jnp.log(x)
  ln_2 = jnp.log(2.0)
  return ln_x / ln_2

print(jax.make_jaxpr(log2_with_print)(3.))

def log2_if_rank_2(x):
  if x.ndim == 2:
    # This branch won't be traced, so there is no JAXPR for it.
    ln_x = jnp.log(x)
    ln_2 = jnp.log(2.0)
    return ln_x / ln_2
  else:
    return x

print(jax.make_jaxpr(log2_if_rank_2)(jax.numpy.array([1, 2, 3])))
