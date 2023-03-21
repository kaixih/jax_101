import jax

# Condition on value of x.

def f(x):
  if x > 0:
    return x
  else:
    return 2 * x

#f_jit = jax.jit(f)
#f_jit(10)  # Should raise an error. 
f_jit_correct = jax.jit(f, static_argnums=0)
print(f_jit_correct(10))

# While loop conditioned on x and n.

def g(x, n):
  i = 0
  while i < n:
    i += 1
  return x + i

#g_jit = jax.jit(g)
#g_jit(10, 20)  # Should raise an error. 
g_jit_correct = jax.jit(g, static_argnames=['n'])
print(g_jit_correct(10, 20))

from functools import partial

@partial(jax.jit, static_argnames=['n'])
def g_jit_decorated(x, n):
  i = 0
  while i < n:
    i += 1
  return x + i

print(g_jit_decorated(10, 20))

# While loop conditioned on x and n with a jitted body.

@jax.jit
def loop_body(prev_i):
  return prev_i + 1

def g_inner_jitted(x, n):
  i = 0
  while i < n:
    i = loop_body(i)
  return x + i

print(g_inner_jitted(10, 20))
