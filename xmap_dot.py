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

def dot_wrapper(x, w):
  return lax.pdot(x, w, 'inputs')

in_axes = (
    {0 : 'batch', 1 :'inputs'},
    {0 : 'inputs', 1 :'outputs'},
)
out_axes = ['batch', 'outputs', ...]

mode = 4
if mode == Mode.DP.value:
  axis_resources = {'batch' : 'x'}
elif mode == Mode.TP_COL.value:
  axis_resources = {'outputs' : 'y'}
elif mode == Mode.TP_ROW.value:
  axis_resources = {'inputs' : 'y'}
elif mode == Mode.DP_TP_COL.value:
  axis_resources = {'batch' : 'x', 'outputs' : 'y'}
if mode == Mode.DP_TP_ROW.value:
  axis_resources = {'batch' : 'x', 'inputs' : 'y'}

func = xmap(dot_wrapper, in_axes=in_axes, out_axes=out_axes,
            axis_resources=axis_resources)

x = jax.random.normal(jax.random.PRNGKey(0), (8192, 784))
w = jax.random.normal(jax.random.PRNGKey(0), (784, 8192))

devices = np.array(jax.local_devices()).reshape((4, 2))
with Mesh(devices, ('x', 'y')):
  if is_show_ir:
    show_ir(func, x, w)
  y = func(x, w)

print("X in shape:", x.shape)
print("W in shape:", w.shape)
print("Y out shape:", y.shape)

