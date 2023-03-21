from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
import jax

sharding = PositionalSharding(mesh_utils.create_device_mesh((8,)))
x = jax.random.normal(jax.random.PRNGKey(0), (8192, 8192))
x = jax.device_put(x, sharding.reshape(4, 2))

@jax.jit
def f(x):
  x = x + 1
  y = jax.lax.with_sharding_constraint(x, sharding.reshape(2, 4))
  return y

lowered = f.lower(x)
print(lowered.as_text())
compiled = lowered.compile()
print(compiled.as_text())

jax.debug.visualize_array_sharding(x)
y = f(x)
jax.debug.visualize_array_sharding(y)
