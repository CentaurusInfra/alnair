"""
# Introduction

Much of the code is inspired by the original implementation by GitHub user [bmild](https://github.com/bmild/nerf) as well as PyTorch implementations from GitHub users [yenchenlin](https://github.com/bmild/nerf), [krrish94](https://github.com/krrish94/nerf-pytorch/) and the Medium Article (https://towardsdatascience.com/its-nerf-from-nothing-build-a-vanilla-nerf-with-pytorch-7846e4c45666). The code has been modified for correctness, clarity, and consistency.
"""

from faulthandler import disable
import os
from typing import Optional, Tuple, List, Union, Callable
import argparse
import warnings
import numpy as np
import torch
import torchvision
from torch import nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import trange
from datetime import datetime
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
#from torch.utils.tensorboard import SummaryWriter

# For repeatability
seed = 3407
torch.manual_seed(seed)
np.random.seed(seed)


"""# Train Hyperparameters

All hyperparameters for training are set here. Defaults were taken from the original, unless computational constraints prohibit them. In this case, we apply sensible defaults that are well within the resources provided by Google Colab.
"""

# Encoders
d_input = 3           # Number of input dimensions
n_freqs = 10          # Number of encoding functions for samples
log_space = True      # If set, frequencies scale in log space
use_viewdirs = True   # If set, use view direction as input
n_freqs_views = 4     # Number of encoding functions for views

# Stratified sampling
n_samples = 64         # Number of spatial samples per ray
perturb = True         # If set, applies noise to sample positions
inverse_depth = False  # If set, samples points linearly in inverse depth

# Model
d_filter = 128          # Dimensions of linear layer filters
n_layers = 2            # Number of layers in network bottleneck
skip = []               # Layers at which to apply input residual
use_fine_model = True   # If set, creates a fine model
d_filter_fine = 128     # Dimensions of linear layer filters of fine network
n_layers_fine = 6       # Number of layers in fine network bottleneck

# Hierarchical sampling
n_samples_hierarchical = 64   # Number of samples per ray
perturb_hierarchical = False  # If set, applies noise to sample positions

# Optimizer
lr = 5e-4  # Learning rate

# Training
n_iters = 10000
batch_size = 2**14          # Number of rays per gradient step (power of 2)
one_image_per_step = True   # One image per gradient step (disables batching)
# TODO: optimze its size to maximize the speed
chunksize = 2**14          # Modify as needed to fit in GPU memory
center_crop = True          # Crop the center of image (one_image_per_)
center_crop_iters = 50      # Stop cropping center after this many epochs
display_rate = 100          # Display test output every X epochs

# Early Stopping
warmup_iters = 100          # Number of iterations during warmup phase
warmup_min_fitness = 10.0   # Min val PSNR to continue training at warmup_iters
n_restarts = 10             # Number of times to restart if training stalls

# We bundle the kwargs for various functions to pass all at once.
kwargs_sample_stratified = {
    'n_samples': n_samples,
    'perturb': perturb,
    'inverse_depth': inverse_depth
}
kwargs_sample_hierarchical = {
    'perturb': perturb
}


"""# Inputs

## Data
First we load the data which we will train our NeRF model on. This is the Lego bulldozer commonly seen in the NeRF demonstrations and serves as a sort of "Hello World" for training NeRFs. Covering other datasets is outside the scope of this code, but feel free to try others included in the original [NeRF source code](https://github.com/bmild/nerf) or your own datasets.
"""

if not os.path.exists('tiny_nerf_data.npz'):
    import requests
    url = 'https://people.eecs.berkeley.edu/~bmild/nerf/tiny_nerf_data.npz'
    r = requests.get(url, allow_redirects=True)

    open('tiny_nerf_data.npz', 'wb').write(r.content)

"""This dataset consists of 106 images taken of the synthetic Lego bulldozer along with poses and a common focal length value. Like the original, we reserve the first 100 images for training and a single test image for validation."""

data = np.load('tiny_nerf_data.npz')
images = data['images']
poses = data['poses']
focal = data['focal']


height, width = images.shape[1:3]
near, far = 2., 6.

n_training = 100
testimg_idx = 101
testimg, testpose = images[testimg_idx], poses[testimg_idx]

"""## Origins and Directions

Recall that NeRF processes inputs from a field of positions (x,y,z) and view directions (θ,φ). To gather these input points, we need to apply inverse rendering to the input images. More concretely, we draw projection lines through each pixel and across the 3D space, from which we can draw samples.

To sample points from the 3D space beyond our image, we first start from the initial pose of every camera taken in the photo set. With some vector math, we can convert these 4x4 pose matrices into a 3D coordinate denoting the origin and a 3D vector indicating the direction. The two together describe a vector that indicates where a camera was pointing when the photo was taken.

The code in the cell below illustrates this by drawing arrows that depict the origin and the direction of every frame.
"""

dirs = np.stack([np.sum([0, 0, -1] * pose[:3, :3], axis=-1) for pose in poses])
origins = poses[:, :3, -1]


def cast_to_image(tensor):  
    r"""
    # This utility function is used to change tensor to image.
    """
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    # Conver to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    # Map back to shape (3, H, W), as tensorboard needs channels first.
    img = np.moveaxis(img, [-1], [0])
    return img

def get_rays(
  height: int,
  width: int,
  focal_length: float,
  c2w: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
  r"""
  Find origin and direction of rays through every pixel and camera origin.
  """

  # Apply pinhole camera model to gather directions at each pixel
  i, j = torch.meshgrid(
      torch.arange(width, dtype=torch.float32).to(c2w),
      torch.arange(height, dtype=torch.float32).to(c2w),
      indexing='ij')
  i, j = i.transpose(-1, -2), j.transpose(-1, -2)
  directions = torch.stack([(i - width * .5) / focal_length,
                            -(j - height * .5) / focal_length,
                            -torch.ones_like(i)
                           ], dim=-1)

  # Apply camera pose to directions
  rays_d = torch.sum(directions[..., None, :] * c2w[:3, :3], dim=-1)

  # Origin is same for all directions (the optical center)
  rays_o = c2w[:3, -1].expand(rays_d.shape)
  return rays_o, rays_d



"""# Architecture

## Stratified Sampling

Now that we have these lines, defined as origin and direction vectors, we can begin the process of sampling them. Recall that NeRF takes a coarse-to-fine sampling strategy, starting with the stratified sampling approach.

The stratified sampling approach splits the ray into evenly-spaced bins and randomly samples within each bin. The `perturb` setting determines whether to sample points uniformly from each bin or to simply use the bin center as the point. In most cases, we want to keep `perturb = True` as it will encourage the network to learn over a continuously sampled space. It may be useful to disable for debugging.
"""

def sample_stratified(
  rays_o: torch.Tensor,
  rays_d: torch.Tensor,
  near: float,
  far: float,
  n_samples: int,
  perturb: Optional[bool] = True,
  inverse_depth: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
  r"""
  Sample along ray from regularly-spaced bins.
  """

  # Grab samples for space integration along ray
  t_vals = torch.linspace(0., 1., n_samples, device=rays_o.device)
  if not inverse_depth:
    # Sample linearly between `near` and `far`
    z_vals = near * (1.-t_vals) + far * (t_vals)
  else:
    # Sample linearly in inverse depth (disparity)
    z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

  # Draw uniform samples from bins along ray
  if perturb:
    mids = .5 * (z_vals[1:] + z_vals[:-1])
    upper = torch.concat([mids, z_vals[-1:]], dim=-1)
    lower = torch.concat([z_vals[:1], mids], dim=-1)
    t_rand = torch.rand([n_samples], device=z_vals.device)
    z_vals = lower + (upper - lower) * t_rand
  z_vals = z_vals.expand(list(rays_o.shape[:-1]) + [n_samples])

  # Apply scale from `rays_d` and offset from `rays_o` to samples
  # pts: (width, height, n_samples, 3)
  pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
  return pts, z_vals


"""## Positional Encoder

Much like Transformers, NeRFs make use of positional encoders. In this case, it's to map the inputs to a higher frequency space to compensate for the bias that neural networks have for learning lower-frequency functions.

Here we build a simple `torch.nn.Module` of our positional encoder. The same encoder implementation can be applied to both input samples and view directions. However, we choose different parameters for these inputs. We use the default settings from the original.
"""

class PositionalEncoder(nn.Module):
  r"""
  Sine-cosine positional encoder for input points.
  """
  def __init__(
    self,
    d_input: int,
    n_freqs: int,
    log_space: bool = False
  ):
    super().__init__()
    self.d_input = d_input
    self.n_freqs = n_freqs
    self.log_space = log_space
    self.d_output = d_input * (1 + 2 * self.n_freqs)
    self.embed_fns = [lambda x: x]

    # Define frequencies in either linear or log scale
    if self.log_space:
      freq_bands = 2.**torch.linspace(0., self.n_freqs - 1, self.n_freqs)
    else:
      freq_bands = torch.linspace(2.**0., 2.**(self.n_freqs - 1), self.n_freqs)

    # Alternate sin and cos
    for freq in freq_bands:
      self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
      self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))
  
  def forward(
    self,
    x
  ) -> torch.Tensor:
    r"""
    Apply positional encoding to input.
    """
    return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)


"""## NeRF Model

Here we define the NeRF model, which consists primarily of a `ModuleList` of `Linear` layers, separated by non-linear activation functions and the occasional residual connection. This model features an optional input for view directions, which will alter the model architecture if provided at instantiation. This implementation is based on Section 3 of the original "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" paper and uses the same defaults.
"""

class NeRF(nn.Module):
  r"""
  Neural radiance fields module.
  """
  def __init__(
    self,
    d_input: int = 3,
    n_layers: int = 8,
    d_filter: int = 256,
    skip: Tuple[int] = (4,),
    d_viewdirs: Optional[int] = None
  ):
    super().__init__()
    self.d_input = d_input
    self.skip = skip
    self.act = nn.functional.relu
    self.d_viewdirs = d_viewdirs

    # Create model layers
    self.layers = nn.ModuleList(
      [nn.Linear(self.d_input, d_filter)] +
      [nn.Linear(d_filter + self.d_input, d_filter) if i in skip \
       else nn.Linear(d_filter, d_filter) for i in range(n_layers - 1)]
    )

    # Bottleneck layers
    if self.d_viewdirs is not None:
      # If using viewdirs, split alpha and RGB
      self.alpha_out = nn.Linear(d_filter, 1)
      self.rgb_filters = nn.Linear(d_filter, d_filter)
      self.branch = nn.Linear(d_filter + self.d_viewdirs, d_filter // 2)
      self.output = nn.Linear(d_filter // 2, 3)
    else:
      # If no viewdirs, use simpler output
      self.output = nn.Linear(d_filter, 4)
  
  def forward(
    self,
    x: torch.Tensor,
    viewdirs: Optional[torch.Tensor] = None
  ) -> torch.Tensor:
    r"""
    Forward pass with optional view direction.
    """

    # Cannot use viewdirs if instantiated with d_viewdirs = None
    if self.d_viewdirs is None and viewdirs is not None:
      raise ValueError('Cannot input x_direction if d_viewdirs was not given.')

    # Apply forward pass up to bottleneck
    x_input = x
    for i, layer in enumerate(self.layers):
      x = self.act(layer(x))
      if i in self.skip:
        x = torch.cat([x, x_input], dim=-1)

    # Apply bottleneck
    if self.d_viewdirs is not None:
      # Split alpha from network output
      alpha = self.alpha_out(x)

      # Pass through bottleneck to get RGB
      x = self.rgb_filters(x)
      x = torch.concat([x, viewdirs], dim=-1)
      x = self.act(self.branch(x))
      x = self.output(x)

      # Concatenate alphas to output
      x = torch.concat([x, alpha], dim=-1)
    else:
      # Simple output
      x = self.output(x)
    return x

"""## Volume Rendering

From the raw NeRF outputs, we still need to convert these into an image. This is where we apply the volume integration described in Equations 1-3 in Section 4 of the paper. Essentially, we take the weighted sum of all samples along the ray of each pixel to get the estimated color value at that pixel. Each RGB sample is weighted by its alpha value. Higher alpha values indicate higher likelihood that the sampled area is opaque, therefore points further along the ray are likelier to be occluded. The cumulative product ensures that those further points are dampened.
"""

def cumprod_exclusive(
  tensor: torch.Tensor
) -> torch.Tensor:
  r"""
  (Courtesy of https://github.com/krrish94/nerf-pytorch)

  Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.

  Args:
  tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
    is to be computed.
  Returns:
  cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
    tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
  """

  # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
  cumprod = torch.cumprod(tensor, -1)
  # "Roll" the elements along dimension 'dim' by 1 element.
  cumprod = torch.roll(cumprod, 1, -1)
  # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
  cumprod[..., 0] = 1.
  
  return cumprod

def raw2outputs(
  raw: torch.Tensor,
  z_vals: torch.Tensor,
  rays_d: torch.Tensor,
  raw_noise_std: float = 0.0,
  white_bkgd: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  r"""
  Convert the raw NeRF output into RGB and other maps.
  """

  # Difference between consecutive elements of `z_vals`. [n_rays, n_samples]
  dists = z_vals[..., 1:] - z_vals[..., :-1]
  dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1)

  # Multiply each distance by the norm of its corresponding direction ray
  # to convert to real world distance (accounts for non-unit directions).
  dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

  # Add noise to model's predictions for density. Can be used to 
  # regularize network during training (prevents floater artifacts).
  noise = 0.
  if raw_noise_std > 0.:
    noise = torch.randn(raw[..., 3].shape) * raw_noise_std

  # Predict density of each sample along each ray. Higher values imply
  # higher likelihood of being absorbed at this point. [n_rays, n_samples]
  alpha = 1.0 - torch.exp(-nn.functional.relu(raw[..., 3] + noise) * dists)

  # Compute weight for RGB of each sample along each ray. [n_rays, n_samples]
  # The higher the alpha, the lower subsequent weights are driven.
  weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)

  # Compute weighted RGB map.
  rgb = torch.sigmoid(raw[..., :3])  # [n_rays, n_samples, 3]
  rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # [n_rays, 3]

  # Estimated depth map is predicted distance.
  depth_map = torch.sum(weights * z_vals, dim=-1)

  # Disparity map is inverse depth.
  disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map),
                            depth_map / torch.sum(weights, -1))

  # Sum of weights along each ray. In [0, 1] up to numerical error.
  acc_map = torch.sum(weights, dim=-1)

  # To composite onto a white background, use the accumulated alpha map.
  if white_bkgd:
    rgb_map = rgb_map + (1. - acc_map[..., None])

  return rgb_map, depth_map, acc_map, weights

"""## Hierarchical Volume Sampling
The 3D space is in fact very sparse with occlusions and so most points don't contribute much to the rendered image. It is therefore more beneficial to oversample regions with a high likelihood of contributing to the integral. Here we apply learned, normalized weights to the first set of samples to create a PDF across the ray, then apply inverse transform sampling to this PDF to gather a second set of samples.
"""

def sample_pdf(
  bins: torch.Tensor,
  weights: torch.Tensor,
  n_samples: int,
  perturb: bool = False
) -> torch.Tensor:
  r"""
  Apply inverse transform sampling to a weighted set of points.
  """

  # Normalize weights to get PDF.
  pdf = (weights + 1e-5) / torch.sum(weights + 1e-5, -1, keepdims=True) # [n_rays, weights.shape[-1]]

  # Convert PDF to CDF.
  cdf = torch.cumsum(pdf, dim=-1) # [n_rays, weights.shape[-1]]
  cdf = torch.concat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1) # [n_rays, weights.shape[-1] + 1]

  # Take sample positions to grab from CDF. Linear when perturb == 0.
  if not perturb:
    u = torch.linspace(0., 1., n_samples, device=cdf.device)
    u = u.expand(list(cdf.shape[:-1]) + [n_samples]) # [n_rays, n_samples]
  else:
    u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=cdf.device) # [n_rays, n_samples]

  # Find indices along CDF where values in u would be placed.
  u = u.contiguous() # Returns contiguous tensor with same values.
  inds = torch.searchsorted(cdf, u, right=True) # [n_rays, n_samples]

  # Clamp indices that are out of bounds.
  below = torch.clamp(inds - 1, min=0)
  above = torch.clamp(inds, max=cdf.shape[-1] - 1)
  inds_g = torch.stack([below, above], dim=-1) # [n_rays, n_samples, 2]

  # Sample from cdf and the corresponding bin centers.
  matched_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]]
  cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), dim=-1,
                       index=inds_g)
  bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), dim=-1,
                        index=inds_g)

  # Convert samples to ray length.
  denom = (cdf_g[..., 1] - cdf_g[..., 0])
  denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
  t = (u - cdf_g[..., 0]) / denom
  samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

  return samples # [n_rays, n_samples]

def sample_hierarchical(
  rays_o: torch.Tensor,
  rays_d: torch.Tensor,
  z_vals: torch.Tensor,
  weights: torch.Tensor,
  n_samples: int,
  perturb: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  r"""
  Apply hierarchical sampling to the rays.
  """

  # Draw samples from PDF using z_vals as bins and weights as probabilities.
  z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
  new_z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], n_samples,
                          perturb=perturb)
  new_z_samples = new_z_samples.detach()

  # Resample points from ray based on PDF.
  z_vals_combined, _ = torch.sort(torch.cat([z_vals, new_z_samples], dim=-1), dim=-1)
  pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_combined[..., :, None]  # [N_rays, N_samples + n_samples, 3]
  return pts, z_vals_combined, new_z_samples

"""## Full Forward Pass

Here is where we put everything together to compute a single forward pass through our model.

Due to potential memory issues, the forward pass is computed in "chunks," which are then aggregated across a single batch. The gradient propagation is done after the whole batch is processed, hence the distinction between "chunks" and "batches." Chunking is especially important for the Google Colab environment, which provides more modest resources than those cited in the original paper.
"""

def get_chunks(
  inputs: torch.Tensor,
  chunksize: int = 2**15
) -> List[torch.Tensor]:
  r"""
  Divide an input into chunks.
  """
  #print("chunksize:", chunksize)
  return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]

def prepare_chunks(
  points: torch.Tensor,
  encoding_function: Callable[[torch.Tensor], torch.Tensor],
  chunksize: int = 2**15
) -> List[torch.Tensor]:
  r"""
  Encode and chunkify points to prepare for NeRF model.
  """
  points = points.reshape((-1, 3))
  points = encoding_function(points)
  points = get_chunks(points, chunksize=chunksize)
  return points

def prepare_viewdirs_chunks(
  points: torch.Tensor,
  rays_d: torch.Tensor,
  encoding_function: Callable[[torch.Tensor], torch.Tensor],
  chunksize: int = 2**15
) -> List[torch.Tensor]:
  r"""
  Encode and chunkify viewdirs to prepare for NeRF model.
  """
  # Prepare the viewdirs
  viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
  viewdirs = viewdirs[:, None, ...].expand(points.shape).reshape((-1, 3))
  viewdirs = encoding_function(viewdirs)
  viewdirs = get_chunks(viewdirs, chunksize=chunksize)
  return viewdirs

def nerf_forward(
  rays_o: torch.Tensor,
  rays_d: torch.Tensor,
  near: float,
  far: float,
  encoding_fn: Callable[[torch.Tensor], torch.Tensor],
  coarse_model: nn.Module,
  kwargs_sample_stratified: dict = None,
  n_samples_hierarchical: int = 0,
  kwargs_sample_hierarchical: dict = None,
  fine_model = None,
  viewdirs_encoding_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
  chunksize: int = 2**15
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
  r"""
  Compute forward pass through model(s).
  """

  # Set no kwargs if none are given.
  if kwargs_sample_stratified is None:
    kwargs_sample_stratified = {}
  if kwargs_sample_hierarchical is None:
    kwargs_sample_hierarchical = {}
  
  # Sample query points along each ray.
  query_points, z_vals = sample_stratified(
      rays_o, rays_d, near, far, **kwargs_sample_stratified)
  

  # Prepare batches.
  batches = prepare_chunks(query_points, encoding_fn, chunksize=chunksize)

  if viewdirs_encoding_fn is not None:
    batches_viewdirs = prepare_viewdirs_chunks(query_points, rays_d,
                                               viewdirs_encoding_fn,
                                               chunksize=chunksize)
  else:
    batches_viewdirs = [None] * len(batches)

  # Coarse model pass.
  # Split the encoded points into "chunks", run the model on all chunks, and
  # concatenate the results (to avoid out-of-memory issues).
  predictions = []
  for batch, batch_viewdirs in zip(batches, batches_viewdirs):
    predictions.append(coarse_model(batch, viewdirs=batch_viewdirs))
  raw = torch.cat(predictions, dim=0)
  raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])


  # Perform differentiable volume rendering to re-synthesize the RGB image.
  rgb_map, depth_map, acc_map, weights = raw2outputs(raw, z_vals, rays_d)
  # rgb_map, depth_map, acc_map, weights = render_volume_density(raw, rays_o, z_vals)
  outputs = {
      'z_vals_stratified': z_vals
  }
  
 
  # Fine model pass.
  if n_samples_hierarchical > 0:
    # Save previous outputs to return.
    rgb_map_0, depth_map_0, acc_map_0 = rgb_map, depth_map, acc_map

    # Apply hierarchical sampling for fine query points.
    query_points, z_vals_combined, z_hierarch = sample_hierarchical(
      rays_o, rays_d, z_vals, weights, n_samples_hierarchical,
      **kwargs_sample_hierarchical)

    # Prepare inputs as before.
    batches = prepare_chunks(query_points, encoding_fn, chunksize=chunksize)
    if viewdirs_encoding_fn is not None:
      batches_viewdirs = prepare_viewdirs_chunks(query_points, rays_d,
                                                 viewdirs_encoding_fn,
                                                 chunksize=chunksize)
    else:
      batches_viewdirs = [None] * len(batches)

    # Forward pass new samples through fine model.
    fine_model = fine_model if fine_model is not None else coarse_model
    predictions = []
    for batch, batch_viewdirs in zip(batches, batches_viewdirs):
      predictions.append(fine_model(batch, viewdirs=batch_viewdirs))
    raw = torch.cat(predictions, dim=0)
    raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

    # Perform differentiable volume rendering to re-synthesize the RGB image.
    rgb_map, depth_map, acc_map, weights = raw2outputs(raw, z_vals_combined, rays_d)
    

    # Store outputs.
    outputs['z_vals_hierarchical'] = z_hierarch
    outputs['rgb_map_0'] = rgb_map_0
    outputs['depth_map_0'] = depth_map_0
    outputs['acc_map_0'] = acc_map_0

  # Store outputs.
  outputs['rgb_map'] = rgb_map
  outputs['depth_map'] = depth_map
  outputs['acc_map'] = acc_map
  outputs['weights'] = weights
  return outputs



"""## Training Classes and Functions

Here we create some helper functions for training. NeRF can be prone to local minima, in which training will quickly stall and produce blank outputs. `EarlyStopping` is used to restart the training when learning stalls, if necessary.
"""

def plot_samples(
  z_vals: torch.Tensor,
  z_hierarch: Optional[torch.Tensor] = None,
  ax: Optional[np.ndarray] = None):
  r"""
  Plot stratified and (optional) hierarchical samples.
  """
  y_vals = 1 + np.zeros_like(z_vals)

  if ax is None:
    ax = plt.subplot()
  ax.plot(z_vals, y_vals, 'b-o')
  if z_hierarch is not None:
    y_hierarch = np.zeros_like(z_hierarch)
    ax.plot(z_hierarch, y_hierarch, 'r-o')
  ax.set_ylim([-1, 2])
  ax.set_title('Stratified  Samples (blue) and Hierarchical Samples (red)')
  ax.axes.yaxis.set_visible(False)
  ax.grid(True)
  return ax

def crop_center(
  img: torch.Tensor,
  frac: float = 0.5
) -> torch.Tensor:
  r"""
  Crop center square from image.
  """
  h_offset = round(img.shape[0] * (frac / 2))
  w_offset = round(img.shape[1] * (frac / 2))
  return img[h_offset:-h_offset, w_offset:-w_offset]

class EarlyStopping:
  r"""
  Early stopping helper based on fitness criterion.
  """
  def __init__(
    self,
    patience: int = 30,
    margin: float = 1e-4
  ):
    self.best_fitness = 0.0  # In our case PSNR
    self.best_iter = 0
    self.margin = margin
    self.patience = patience or float('inf')  # epochs to wait after fitness stops improving to stop

  def __call__(
    self,
    iter: int,
    fitness: float
  ):
    r"""
    Check if criterion for stopping is met.
    """
    if (fitness - self.best_fitness) > self.margin:
      self.best_iter = iter
      self.best_fitness = fitness
    delta = iter - self.best_iter
    stop = delta >= self.patience  # stop training if patience exceeded
    return stop

def init_models(gpu):
  r"""
  Initialize models, encoders, and optimizer for NeRF training.
  """
  # Encoders
  encoder = PositionalEncoder(d_input, n_freqs, log_space=log_space)
  encode = lambda x: encoder(x)

  # View direction encoders
  if use_viewdirs:
    encoder_viewdirs = PositionalEncoder(d_input, n_freqs_views,
                                        log_space=log_space)
    encode_viewdirs = lambda x: encoder_viewdirs(x)
    d_viewdirs = encoder_viewdirs.d_output
  else:
    encode_viewdirs = None
    d_viewdirs = None

  # Models
  model = NeRF(encoder.d_output, n_layers=n_layers, d_filter=d_filter, skip=skip,
              d_viewdirs=d_viewdirs)
  model.to(gpu)
  ddp_model = DDP(model, device_ids=[gpu])

  model_params = list(ddp_model.parameters())
  if use_fine_model:
    fine_model = NeRF(encoder.d_output, n_layers=n_layers, d_filter=d_filter, skip=skip,
                      d_viewdirs=d_viewdirs)
    fine_model.to(gpu)
    ddp_fine_model = DDP(fine_model, device_ids=[gpu])
    model_params = model_params + list(ddp_fine_model.parameters())
  else:
    fine_model = None

  # Optimizer
  optimizer = torch.optim.Adam(model_params, lr=lr)

  # Early Stopping
  warmup_stopper = EarlyStopping(patience=50)

  return ddp_model, ddp_fine_model, encode, encode_viewdirs, optimizer, warmup_stopper

"""## Training Loop

Here we start training our model. 
"""

def train(gpu, args):
  r"""
  Launch training session for NeRF.
  """
  # Shuffle rays across all images.
  gpu_list = [int(gpu) for gpu in args.gpus.split(',')]
  rank = sum(gpu_list[:args.id]) + gpu
  dist.init_process_group(backend='gloo', init_method='env://', world_size=args.world_size, rank=rank)
  #print("my process rank:{} on gpu {}\n".format(rank, gpu))
  model, fine_model, encode, encode_viewdirs, optimizer, warmup_stopper = init_models(gpu)

# Gather as torch tensors
  images = torch.from_numpy(data['images'][:n_training]).to(gpu)
  poses = torch.from_numpy(data['poses']).to(gpu)
  focal = torch.from_numpy(data['focal']).to(gpu)
  testimg = torch.from_numpy(data['images'][testimg_idx]).to(gpu)
  testpose = torch.from_numpy(data['poses'][testimg_idx]).to(gpu)

  #logdir = "runs/nerf_ddp/" + datetime.now().strftime("%Y%m%d-%H%M%S")
  #if rank == 0:
  #  writer = SummaryWriter(logdir, disabled=True)


  if not one_image_per_step:
      height, width = images.shape[1:3]
      all_rays = torch.stack([torch.stack(get_rays(height, width, focal, p, gpu), 0)
                          for p in poses[:n_training]], 0)
      rays_rgb = torch.cat([all_rays, images[:, None]], 1)
      rays_rgb = torch.permute(rays_rgb, [0, 2, 3, 1, 4])
      rays_rgb = rays_rgb.reshape([-1, 3, 3])
      rays_rgb = rays_rgb.type(torch.float32)
      rays_rgb = rays_rgb[torch.randperm(rays_rgb.shape[0])]
      i_batch = 0

  train_psnrs = []
  val_psnrs = []
  iternums = []
  start = datetime.now()

  for i in trange(n_iters):
    model.train()
    if one_image_per_step:
        # Randomly pick an image as the target.
        target_img_idx = np.random.randint(images.shape[0])
        target_img = images[target_img_idx].to(gpu)
        ## TODO: test it later
        #  if center_crop and i < center_crop_iters:
        #     target_img = crop_center(target_img)
        height, width = target_img.shape[:2]


        target_pose = poses[target_img_idx].to(gpu)
        rays_o, rays_d = get_rays(height, width, focal, target_pose)
        rays_o = rays_o.reshape([-1, 3])
        rays_d = rays_d.reshape([-1, 3])

        # use to partition data 
        partitions = list(range(0, rays_o.shape[0], int(rays_o.shape[0]/(args.world_size))))
        partitions.append(rays_o.shape[0])
        rays_o_ddp = rays_o[partitions[rank]: partitions[rank+1]]
        rays_d_ddp = rays_d[partitions[rank]: partitions[rank+1]]

    else:
    # Random over all images.
        batch = rays_rgb[i_batch:i_batch + batch_size]
        batch = torch.transpose(batch, 0, 1)
        rays_o, rays_d, target_img = batch
        height, width = target_img.shape[:2]
        i_batch += batch_size
        # Shuffle after one epoch
        if i_batch >= rays_rgb.shape[0]:
            rays_rgb = rays_rgb[torch.randperm(rays_rgb.shape[0])]
            i_batch = 0

    target_img = target_img.reshape([-1, 3])

    # Run one iteration of TinyNeRF and get the rendered RGB image.
    outputs = nerf_forward(rays_o_ddp, rays_d_ddp,
                        near, far, encode, model,
                        kwargs_sample_stratified=kwargs_sample_stratified,
                        n_samples_hierarchical=n_samples_hierarchical,
                        kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                        fine_model=fine_model,
                        viewdirs_encoding_fn=encode_viewdirs,
                        chunksize=chunksize)
    
      # Check for any numerical issues.
    for k, v in outputs.items():
        if torch.isnan(v).any():
            print(f"! [Numerical Alert] {k} contains NaN.")
        if torch.isinf(v).any():
            print(f"! [Numerical Alert] {k} contains Inf.")

    rgb_predicted = outputs['rgb_map']
    rgb_predicted_coarse = outputs['rgb_map_0']

    # #visualize testing
    # if rank == 0 and i % display_rate == 0:
    #   writer.add_image(
    #       "train/rgb_fine", cast_to_image(rgb_predicted.reshape([-1, width, 3])), i
    #   )
    #   writer.add_image(
    #       "train/rgb_coarse", cast_to_image(rgb_predicted_coarse.reshape([-1, width, 3])), i
    #   )
    #   writer.add_image(
    #       "train/rgb_ground truth", cast_to_image(target_img[start_heights[rank]: start_heights[rank+1]].reshape([-1, width, 3])), i
    #   )

    coarse_loss = torch.nn.functional.mse_loss(rgb_predicted_coarse, target_img[partitions[rank]: partitions[rank+1]])
    fine_loss = torch.nn.functional.mse_loss(rgb_predicted, target_img[partitions[rank]: partitions[rank+1]])
    loss = coarse_loss + (fine_loss if fine_loss is not None else 0.0)

    # Backprop!

    loss.backward()
    optimizer.step()
    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
    loss /= args.world_size

    # if rank == 0:
    #   writer.add_scalar("train/fine loss", fine_loss.item(), i)
    #   writer.add_scalar("train/coarse_loss", coarse_loss.item(), i)
    #   writer.add_scalar("train/loss", loss.item(), i)

    optimizer.zero_grad()

    # Compute mean-squared error between predicted and target images.
    psnr = -10. * torch.log10(loss)
    train_psnrs.append(psnr.item())

    # Evaluate testimg at given display rate.
    if rank == 0 and i % display_rate == 0:
        model.eval()
        #use testimg to get consistant curve
        height, width = testimg.shape[:2]
        rays_o, rays_d = get_rays(height, width, focal, testpose)
        rays_o = rays_o.reshape([-1, 3])
        rays_d = rays_d.reshape([-1, 3])
        outputs = nerf_forward(rays_o, rays_d,
                                near, far, encode, model,
                                kwargs_sample_stratified=kwargs_sample_stratified,
                                n_samples_hierarchical=n_samples_hierarchical,
                                kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                                fine_model=fine_model,
                                viewdirs_encoding_fn=encode_viewdirs,
                                chunksize=chunksize)
        rgb_predicted_coarse = outputs['rgb_map_0']
        rgb_predicted = outputs['rgb_map']
        fine_loss = torch.nn.functional.mse_loss(rgb_predicted, testimg.reshape(-1, 3)) 
        coarse_loss = torch.nn.functional.mse_loss(rgb_predicted_coarse, testimg.reshape(-1, 3)) 
        loss = fine_loss + coarse_loss
        
  
        # writer.add_scalar("validation/loss", loss.item(), i)
        # writer.add_scalar("validation/coarse_loss", coarse_loss.item(), i)
        # writer.add_scalar("validation/fine_loss", fine_loss.item(), i)
        # writer.add_image(
        #     "validation/rgb_fine", cast_to_image(rgb_predicted.reshape([-1, width, 3])), i
        # )
        # writer.add_image(
        #     "validation/rgb_coarse", cast_to_image(rgb_predicted_coarse.reshape([-1, width, 3])), i
        # )
        # writer.add_image(
        #     "validation/rgb_ground truth", cast_to_image(testimg.reshape([-1, width, 3])), i
        # )

        # loss = torch.nn.functional.mse_loss(rgb_predicted, testimg.reshape(-1, 3))
        #print("Loss:", loss.item())
        #print('Iteration: {}/{}, Val Loss: {:.4f}, Time: {}'.format(i, n_iters, loss.item(), (datetime.now() - start)))

        val_psnr = -10. * torch.log10(loss)
        
        val_psnrs.append(val_psnr.item())


        # writer.add_scalar("psnr", val_psnr.item(), i)

        iternums.append(i)

        # # Save plot directly
        # fig, ax = plt.subplots(1, 5, figsize=(24,5), gridspec_kw={'width_ratios': [1, 1, 1, 1, 2]})
        # ax[0].imshow(rgb_predicted.reshape([height, width, 3]).detach().cpu().numpy())
        # ax[0].set_title(f'Fine Iteration: {i}')
        # ax[1].imshow(rgb_predicted_coarse.reshape([height, width, 3]).detach().cpu().numpy())
        # ax[1].set_title(f'Coarse Iteration: {i}')
        # ax[2].imshow(testimg.detach().cpu().numpy())
        # ax[2].set_title(f'Target')
        # ax[3].plot(range(0, i + 1), train_psnrs, 'r')
        # ax[3].plot(iternums, val_psnrs, 'b')
        # ax[3].set_title('PSNR (train=red, val=blue')
        # z_vals_strat = outputs['z_vals_stratified'].view((-1, n_samples))
        # z_sample_strat = z_vals_strat[z_vals_strat.shape[0] // 2].detach().cpu().numpy()
        # if 'z_vals_hierarchical' in outputs:
        #     z_vals_hierarch = outputs['z_vals_hierarchical'].view((-1, n_samples_hierarchical))
        #     z_sample_hierarch = z_vals_hierarch[z_vals_hierarch.shape[0] // 2].detach().cpu().numpy()
        # else:
        #     z_sample_hierarch = None
        # _ = plot_samples(z_sample_strat, z_sample_hierarch, ax=ax[4])
        # ax[4].margins(0)
        # #plt.show()
        
        # path = '{}/logs/ddp'.format(os.getcwd())
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # fig.savefig(os.path.join(path, "Iteration_{}.png".format(i)))
        # plt.close(fig)

    # # Check PSNR for issues and stop if any are found.
    # if i == warmup_iters - 1:
    #     if val_psnr < warmup_min_fitness:
    #         print(f'Val PSNR {val_psnr} below warmup_min_fitness {warmup_min_fitness}. Stopping...')
    #         return False, train_psnrs, val_psnrs
    # elif i < warmup_iters:
    #     if warmup_stopper is not None and warmup_stopper(i, psnr):
    #         print(f'Train PSNR flatlined at {psnr} for {warmup_stopper.patience} iters. Stopping...')
    #         return False, train_psnrs, val_psnrs
  if gpu == 0:    
    print("Training complete in: " + str(datetime.now() - start))
    # with open('results.txt', 'a') as f:
    #   f.writelines("Training complete in: " + str(datetime.now() - start) + '\n')

  # # close tensorbloard
  # if rank == 0:
  #   writer.close()
  
  cleanup()
  return True, train_psnrs, val_psnrs


# # TODO: check this later
# Run training session(s)
# for _ in range(n_restarts):
#     model, fine_model, encode, encode_viewdirs, optimizer, warmup_stopper = init_models()
#     success, train_psnrs, val_psnrs = train()
#     if success and val_psnrs[-1] >= warmup_min_fitness:
#         print('Training successful!')
#         break


def cleanup():
    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default="", type=str,
                        help='number of gpus of each node')
    parser.add_argument('-i', '--id', default=0, type=int,
                        help='the id of the node which is determined by the correponding index in the gpu list')                        
    parser.add_argument('-t', '--number_of_tests', default=1, type=int, metavar='N',
                    help='number of tests that you want to run')                                     
    args = parser.parse_args()
    gpu_list = [int(gpu) for gpu in args.gpus.split(',')]
    args.world_size = sum(gpu_list)
    print("This code is running. Check your master IP address if you see nothing after a while.")
    args.world_size = sum(gpu_list)
    # This is the master node's IP
    os.environ["MASTER_ADDR"] = "10.145.83.35"
    os.environ["MASTER_PORT"] = "9515"
    for _ in range(args.number_of_tests):
      mp.spawn(train,
          nprocs=gpu_list[args.id],
          args=(args,))
    print("Done!")

if __name__=="__main__":
    main()







