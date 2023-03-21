from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
import jax
import jax.numpy as jnp

sharding = PositionalSharding(mesh_utils.create_device_mesh((8,)))
x = jax.random.normal(jax.random.PRNGKey(0), (800,))

print("Ref Output:", jnp.max(x))

x = jax.device_put(x, sharding)

@jax.jit
def f(x):
  amax = jnp.max(x)
  # No need to explicitly apply the sharding constraint here, since the above
  # reduction is a kind of "contraction" similar with the dot op and thus an
  # all-reduce collective op will be called at the end.
  #amax = jax.lax.with_sharding_constraint(amax, sharding.replicate())
  return amax

lowered = f.lower(x)
print(lowered.as_text())
compiled = lowered.compile()
print(compiled.as_text())

jax.debug.visualize_array_sharding(x)
y = f(x)
#jax.debug.visualize_array_sharding(y)
print("Output:", y)
