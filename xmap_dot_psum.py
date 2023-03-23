from enum import Enum
from functools import partial

import argparse
import numpy as np

from jax import lax
from jax.experimental.maps import xmap
from jax.sharding import Mesh

import jax
import jax.numpy as jnp

parser = argparse.ArgumentParser(description='Different DP+TP modes')
parser.add_argument('--mode', type=int, help='parallelism', default=1)
parser.add_argument('--show_ir', action='store_true', help='show compiled IR')
args = parser.parse_args()
mode = args.mode
is_show_ir = args.show_ir

class Mode(Enum):
  DP = 1
  TP_COL = 2
  TP_ROW = 3
  DP_TP_COL = 4
  DP_TP_ROW = 5

print("Working on mode:", Mode(mode))

def show_ir(f, x, w):
  lowered = f.lower(x, w)
  print(lowered.as_text())
  compiled = lowered.compile()
  print(compiled.as_text())

def dot_wrapper(x, w, mode):
  y = lax.dot(x, w)
  if mode in (Mode.TP_ROW.value, Mode.DP_TP_ROW.value):
    y = lax.psum(y, axis_name='inputs')
  return y

if mode == Mode.DP.value:
  in_axes = (
      {0 : 'batch'},
      {},
  )
  out_axes = (
      {0 : 'batch'}
  )
  axis_resources = {'batch' : 'x'}
  x_new_shape = (4, 2048, 784)
  w_new_shape = (784, 8192)
elif mode == Mode.TP_COL.value:
  in_axes = (
      {},
      {1 : 'outputs'},
  )
  out_axes = (
      {1 : 'outputs'}
  )
  axis_resources = {'outputs' : 'y'}
  x_new_shape = (8192, 784)
  w_new_shape = (784, 2, 4096)
elif mode == Mode.TP_ROW.value:
  in_axes = (
      {1 : 'inputs'},
      {0 : 'inputs'},
  )
  out_axes = [...]
  axis_resources = {'inputs' : 'y'}
  x_new_shape = (8192, 2, 392)
  w_new_shape = (2, 392, 8192)
elif mode == Mode.DP_TP_COL.value:
  in_axes = (
      {0 : 'batch'},
      {1 : 'outputs'},
  )
  out_axes = (
      {0 : 'batch', 2 : 'outputs'}
  )
  axis_resources = {'batch' : 'x', 'outputs' : 'y'}
  x_new_shape = (4, 2048, 784)
  w_new_shape = (784, 2, 4096)
elif mode == Mode.DP_TP_ROW.value:
  in_axes = (
      {0 : 'batch', 2 : 'inputs'},
      {0 : 'inputs'},
  )
  out_axes = ['batch', ...]
  axis_resources = {'batch' : 'x', 'inputs' : 'y'}
  x_new_shape = (4, 2048, 2, 392)
  w_new_shape = (2, 392, 8192)

dot_func = partial(dot_wrapper, mode=mode)

func = xmap(dot_func, in_axes=in_axes, out_axes=out_axes,
            axis_resources=axis_resources)

x = jax.random.normal(jax.random.PRNGKey(0), (8192, 784))
w = jax.random.normal(jax.random.PRNGKey(0), (784, 8192))

x = x.reshape(x_new_shape)
w = w.reshape(w_new_shape)

devices = np.array(jax.local_devices()).reshape((4, 2))
with Mesh(devices, ('x', 'y')):
  if is_show_ir:
    show_ir(func, x, w)
  y = func(x, w)

print("X in shape:", x.shape)
print("W in shape:", w.shape)
print("Y out shape:", y.shape)

print("Y reshaped out shape:", y.reshape((8192, 8192)).shape)

