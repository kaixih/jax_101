import jax
import jax.numpy as jnp
import numpy as np

def f(x, y):
  return 2 * x + y
x, y = 3, 4

lowered = jax.jit(f).lower(x, y)
print(lowered.as_text())

compiled = lowered.compile()
print(compiled.as_text())

print("JIT results:", jax.jit(f)(x, y))
print("AOT results:", jax.jit(f).lower(x, y).compile()(x, y))

i32_scalar = jax.ShapeDtypeStruct((), jnp.dtype('int32'))
print(jax.jit(f).lower(i32_scalar, i32_scalar).compile()(x, y))

#x_1d = y_1d = jnp.arange(3)
#print(jax.jit(f).lower(i32_scalar, i32_scalar).compile()(x_1d, y_1d))
#x_f = y_f = 72.0
#print(jax.jit(f).lower(i32_scalar, i32_scalar).compile()(x_f, y_f))
