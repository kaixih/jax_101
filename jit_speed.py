import jax
import jax.numpy as jnp
import timeit

def selu(x, alpha=1.67, lambda_=1.05):
  #print("here")
  return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

x = jnp.arange(1000000)

#t = timeit.timeit(lambda: selu(x).block_until_ready(), number=50)
#print("Ref:", t)

selu_jit = jax.jit(selu)

print(selu_jit.lower(x).as_text())

# Warm up
selu_jit(x).block_until_ready()

t = timeit.timeit(lambda: selu_jit(x).block_until_ready(), number=50)
print("JIT:", t)
