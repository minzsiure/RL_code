import torch
import torch.nn as nn

##########AUXILLIARY FUNCTIONS##########

@torch.no_grad
def default_init(tensor, scale=1.0):
  """Initialize a tensor according to https://docs.jax.dev/en/latest/_autosummary/jax.nn.initializers.variance_scaling.html

  Samples drawn from Unif(-a, a) where a = sqrt(scale / n) and n = (fan_in + fan_out)/2

  Args:
    tensor: weights to be initialized
    scale: default 1.0
  
  Returns:
    Does in place initialization of tensor. 
  """
  fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
  fan_avg = (fan_in + fan_out) / 2.0
  limit = (scale / fan_avg)**.5
  tensor.uniform_(-limit, limit)

def get_transformed_mode(transformed_dist):
    """
    Computes the mode of a PyTorch TransformedDistribution.

    Args:
        transformed_dist (torch.distributions.TransformedDistribution): The transformed distribution.
    
    Returns:
        torch.Tensor: The mode of the transformed distribution.
    """
    # Determine the mode of the base distribution.
    # Use the mode() method if it exists; otherwise, fall back to the mean.
    if hasattr(transformed_dist.base_dist, 'mode'):
        base_mode = transformed_dist.base_dist.mode
    else:
        base_mode = transformed_dist.base_dist.mean

    # Apply each transform in sequence to obtain the final mode.
    mode = base_mode
    for transform in transformed_dist.transforms:
        mode = transform(mode)
    return mode

##########ENSEMBLE NETWORKS##########

class Ensemble(nn.Module):
    """
    Ensemble wrapper that creates multiple instances of a module and
    applies them to the same input, stacking the results along a new axis.
    """
    def __init__(self, module_class, num_ensembles, *args, **kwargs):
        super().__init__()
        self.ensemble = nn.ModuleList(
            [module_class(*args, **kwargs) for _ in range(num_ensembles)]
        )
    
    def forward(self, x):
        # Apply each ensemble member to the input.
        outputs = [module(x) for module in self.ensemble]
        # Stack the outputs along a new dimension (for example, axis 0).
        return torch.stack(outputs, dim=0)

##########MLP##########

class MLP(nn.Module):
  """
  Multi-layer perceptron.

  Args:
      hidden_dims (Sequence[int]): List of hidden layer dimensions.
      activations (callable or nn.Module): Activation function (default: GELU).
      activate_final (bool): Whether to apply activation (and layer norm) to the final layer.
      kernel_init (callable, optional): Function to initialize the linear layer weights.
          For example: lambda w: nn.init.xavier_uniform_(w). Default is None, which uses PyTorch's default.
      layer_norm (bool): Whether to apply LayerNorm after the activation.

  Attributes:
      intermediate_feature: The output of the penultimate layer (i.e. after activation and layer norm).
  """
  def __init__(self, hidden_dims, activations=nn.GELU(), activate_final=False, kernel_init=None, layer_norm=False):
    super().__init__()
    # Initialize model parameters
    self.hidden_dims = hidden_dims
    self.activate_final = activate_final
    self.layer_norm = layer_norm
    self.activations = activations
    self.kernel_init = kernel_init
    self.layers = nn.ModuleList()
    self.lns = nn.ModuleList() if self.layer_norm else None
    self.intermediate_features = None    

    # Initialize MLP layers
    for i, dim in enumerate(hidden_dims[1:]):
      lin_layer = nn.Linear(hidden_dims[i], dim)
      if self.kernel_init is not None:
        self.kernel_init(lin_layer.weight)
      self.layers.append(lin_layer)
      if self.layer_norm:
        self.lns.append(nn.LayerNorm(dim))

  def forward(self, x):
    for i, layer in enumerate(self.layers):
      x = layer(x)
      # Apply activation function for everything but the last layer
      # Optionally apply activation function on last layer if designated
      if i < len(self.layers) - 1 or self.activate_final:
        x = self.activations(x)
        # Optionally apply layer norm 
        if self.layer_norm:
          x = self.lns[i](x)
      # Store intermediate values (penultimate layer) 
      if i == len(self.hidden_dims) - 2:
        self.intermediate_features = x
    return x

##########GAUSSIAN ACTOR NET##########

class Actor(nn.Module):
  """
  Gaussian actor network.
    
  Attributes:
      hidden_dims: List of hidden layer dimensions.
      action_dim: Dimension of the action space.
      layer_norm: Whether to apply layer normalization.
      log_std_min: Minimum value for log standard deviation.
      log_std_max: Maximum value for log standard deviation.
      tanh_squash: Whether to squash the action distribution using tanh.
      state_dependent_std: If True, standard deviation is a function of the state.
      const_std: If True and state_dependent_std is False, use constant std (zeros).
      final_fc_init_scale: Scale factor for initializing the final fully-connected layers.
      encoder: Optional encoder module to process the observations.
  """
  def __init__(self,
               action_dim,
               hidden_dims,
               layer_norm=False,
               log_std_min=-5,
               log_std_max=2,
               tanh_squash=False,
               state_dependent_std=False,
               const_std=True,
               final_fc_init_scale=1e-2,
               encoder=None):
    # Initialize Model Parameters
    super().__init__()
    self.action_dim = action_dim
    self.hidden_dims = hidden_dims
    self.layer_norm = layer_norm
    self.log_std_min = log_std_min
    self.log_std_max = log_std_max
    self.tanh_squash = tanh_squash
    self.state_dependent_std = state_dependent_std
    self.const_std = const_std
    self.final_fc_init_scale = final_fc_init_scale
    self.encoder = encoder

    # Initialize networks

    # Actor network - MLP
    self.actor_net = MLP(self.hidden_dims, 
                         activate_final=True,
                         layer_norm=self.layer_norm)
    
    # Mean network - from actor_network output to actions
    self.mean_net = nn.Linear(self.hidden_dims[-1], self.action_dim)
    # Scale the final layer's weights.
    default_init(self.mean_net.weight, scale=self.final_fc_init_scale)

    # State dependent scaling
    if self.state_dependent_std:
      self.log_std_net = nn.Linear(self.hidden_dims[-1], action_dim)
      default_init(self.log_std_net.weight, scale=self.final_fc_init_scale)
    else:
      if not self.const_std:
        # Create a learnable parameter for log standard deviations.
        self.log_stds = nn.Parameter(torch.zeros(self.action_dim))
        # If const_std is True, we will simply use zeros.

  def forward(self, observations, temperature=1.0):
    """
    Forward pass to produce an action distribution.
        
    Args:
      observations: Input observations (tensor).
      temperature: Scaling factor for the standard deviation.
        
    Returns:
      A PyTorch distribution representing the (optionally transformed) Gaussian.
    """
    if self.encoder is not None:
      inputs = self.encoder(observations)
    else:
      inputs = observations
    outputs = self.actor_net(inputs)

    # Action distribution means
    means = self.mean_net(outputs)


    # Action distribution covariances 
    if self.state_dependent_std:
      log_stds = self.log_std_net(outputs)
    else:
      if self.const_std:
        log_stds = torch.zeros_like(means)
      else:
        log_stds = self.log_stds
    
    log_stds = torch.clamp(log_stds, self.log_std_min, self.log_std_max)
    stds = torch.exp(log_stds) * temperature

    base_dist = torch.distributions.Normal(means, stds)
    distribution = torch.distributions.Independent(base_dist, reinterpreted_batch_ndims=1)
    
    # Apply distribution transformation
    if self.tanh_squash:
      transforms = [torch.distributions.transforms.TanhTransform(cache_size=1)]
      distribution = torch.distributions.TransformedDistribution(distribution, transforms)

    return distribution

###########VALUE CRITIC NETWORK##########

class Value(nn.Module):
  """Value/critic network.
    
  This module can be used for both state-value V(s, g) or critic Q(s, a, g) functions.
    
  Attributes:
      hidden_dims: Sequence of hidden layer dimensions (including input dimension).
      layer_norm: Whether to apply layer normalization.
      num_ensembles: Number of ensemble components.
      encoder: Optional encoder module to process observations.
  """
  def __init__(self, input_dim, hidden_dims, layer_norm=True, num_ensembles=2, encoder=None):
    super().__init__()
    self.hidden_dims = hidden_dims
    self.layer_norm = layer_norm
    self.num_ensembles = num_ensembles
    self.encoder = encoder

    # Use MLP as the value network, ensemblize if necessary
    # Make sure the models go through the traditional MLP layers and then output a scalar
    # breakpoint()
    # mlp_dims = tuple(self.hidden_dims + [1])
    mlp_dims = (input_dim,) + tuple(self.hidden_dims) + (1,)

    if self.num_ensembles > 1:
      self.value_net = Ensemble(MLP, num_ensembles=self.num_ensembles, hidden_dims=mlp_dims, activate_final=False, layer_norm=self.layer_norm)
    else:
      self.value_net = MLP(mlp_dims, layer_norm=self.layer_norm, activate_final=False)

  def forward(self, observations, actions=None):
    """Compute value or critic values.
      
    Args:
      observations: Tensor of observations.
      actions: Tensor of actions (optional).
      
    Returns:
      A tensor of scalar values. If using ensembles, the output has an ensemble dimension.
      """
    if self.encoder is not None:
      inputs = [self.encoder(observations)]
    else:
      inputs = [observations]
    if actions is not None:
        inputs.append(actions)
    inputs = torch.cat(inputs, dim=-1)

    v = self.value_net(inputs).squeeze(-1)
    return v


##########ACTOR VECTOR FIELD##########

class ActorVectorField(nn.Module):
  """Actor vector field network for flow matching.

  Args:
    hidden_dims: Hidden layer dimensions
    action_dim: Action dimension
    layer_norm: Whether to apply layer normalization 
    encoder: Optional encoder module to encode the inputs
  """
  def __init__(self, input_dim, hidden_dims, action_dim, layer_norm=False, encoder=None):
    super().__init__()
    self.input_dim = input_dim
    self.hidden_dims = hidden_dims
    self.aciton_dim = action_dim
    self.layer_norm = layer_norm
    self.encoder = encoder

    # mlp_dims = tuple(hidden_dims + [action_dim])
    mlp_dims = (input_dim,) + tuple(hidden_dims) + (action_dim,)
    self.mlp = MLP(mlp_dims, activate_final=False, layer_norm=self.layer_norm)

  def forward(self, observations, actions, times=None, is_encoded=False):
    """Returns the vectors at the given states, actions, and times (optional).

    Args:
      observations: Observations 
      actions: Actions
      times: Times (optional)
      is_encoded: Whether the observations are already encoded.
    """
    if not is_encoded and self.encoder is not None:
      observations = self.encoder(observations)
    if times is None:
      inputs = torch.cat([observations, actions], dim=-1)
    else:
      inputs = torch.cat([observations, actions, times], dim=-1)
    
    v = self.mlp(inputs)
    return v