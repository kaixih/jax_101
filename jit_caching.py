
from functools import partial
import jax
import timeit

def unjitted_loop_body(prev_i):
  return prev_i + 1

def g_inner_jitted_partial(x, n):
  i = 0
  #f = jax.jit(partial(unjitted_loop_body))
  while i < n:
    # Don't do this! each time the partial returns
    # a function with different hash
    i = jax.jit(partial(unjitted_loop_body))(i)
    #i = f(i)
  return x + i

def g_inner_jitted_lambda(x, n):
  i = 0
  #f = jax.jit(lambda x: unjitted_loop_body(x))
  while i < n:
    # Don't do this!, lambda will also return
    # a function with a different hash
    i = jax.jit(lambda x: unjitted_loop_body(x))(i)
    #i = f(i)
  return x + i

def g_inner_jitted_normal(x, n):
  i = 0
  #f = jax.jit(unjitted_loop_body)
  while i < n:
    # this is OK, since JAX can find the
    # cached, compiled function
    i = jax.jit(unjitted_loop_body)(i)
    #i = f(i)
  return x + i

print("jit called in a loop with partials:")
t = timeit.timeit(lambda: g_inner_jitted_partial(10, 20).block_until_ready(), number=50)
print("JIT no cache:", t)

print("jit called in a loop with lambdas:")
t = timeit.timeit(lambda: g_inner_jitted_lambda(10, 20).block_until_ready(), number=50)
print("JIT no cache:", t)

print("jit called in a loop with caching:")
t = timeit.timeit(lambda: g_inner_jitted_normal(10, 20).block_until_ready(), number=50)
print("JIT with cache:", t)
