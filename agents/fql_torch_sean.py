import copy
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
import ml_collections
import torch.optim as optim

from utils.encoders import encoder_modules
from utils.torch_utils import ModuleDict, TrainState
from utils.torch_networks import ActorVectorField, Value

class FQLAgent_Torch:
    """Flow Q-learning (FQL) agent in PyTorch."""

    def __init__(self, rng, network, config):
        self.rng = rng 
        self.network = network  
        self.config = config  

    def critic_loss(self, batch, grad_params=None, rng=None):
        # use current network parameters
        if rng is None:
            rng = torch.Generator()
        with torch.no_grad():
          # sample next actions
          next_actions = self.sample_actions(batch['next_observations'], seed=rng)
          next_actions = torch.clamp(next_actions, -1, 1)

          # compute next Q values using the target critic
          next_qs = self.network.model(**{
              'observations': batch['next_observations'],
              'actions': next_actions
          }, name='target_critic')
          if self.config['q_agg'] == 'min':
              next_q = torch.min(next_qs, dim=0)[0]
          else:
              next_q = torch.mean(next_qs, dim=0)

          target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q

        # compute current Q values using the critic
        q = self.network.model(**{
            'observations': batch['observations'],
            'actions': batch['actions']
        }, name='critic')
        critic_loss = F.mse_loss(q, target_q)

        info = {
            'critic_loss': critic_loss.item(),
            'q_mean': q.mean().item(),
            'q_max': q.max().item(),
            'q_min': q.min().item(),
        }
        return critic_loss, info

    def actor_loss(self, batch, grad_params=None, rng=None):
        if rng is None:
            rng = torch.Generator()
        batch_size, action_dim = batch['actions'].shape

        # BC flow loss
        x_0 = torch.randn(batch_size, action_dim, generator=rng)
        x_1 = batch['actions']
        t = torch.rand(batch_size, 1, generator=rng)
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.model(**{
            'observations': batch['observations'],
            'actions': x_t,
            'times': t
        }, name='actor_bc_flow')
        bc_flow_loss = F.mse_loss(pred, vel)

        # distillation loss
        noises = torch.randn(batch_size, action_dim, generator=rng)
        target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises)
        actor_actions = self.network.model(**{
            'observations': batch['observations'],
            'actions': noises
        }, name='actor_onestep_flow')
        distill_loss = F.mse_loss(actor_actions, target_flow_actions)

        # Q loss
        actor_actions_clipped = torch.clamp(actor_actions, -1, 1)
        qs = self.network.model(**{
            'observations': batch['observations'],
            'actions': actor_actions_clipped
        }, name='critic')
        q_val = torch.mean(qs, dim=0)
        q_loss = -q_val.mean()
        if self.config['normalize_q_loss']:
            lam = 1 / torch.abs(q_val).mean().detach()
            q_loss = lam * q_loss

        actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss

        # additional metrics
        actions = self.sample_actions(batch['observations'], seed=rng)
        mse = F.mse_loss(actions, batch['actions'])

        info = {
            'actor_loss': actor_loss.item(),
            'bc_flow_loss': bc_flow_loss.item(),
            'distill_loss': distill_loss.item(),
            'q_loss': q_loss.item(),
            'q': q_val.mean().item(),
            'mse': mse.item(),
        }
        return actor_loss, info

    def total_loss(self, batch, grad_params=None, rng=None):
        info = {}
        if rng is None:
            rng = torch.Generator()

        critic_loss, critic_info = self.critic_loss(batch, grad_params, rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    @torch.no_grad
    def target_update(self, module_name):
        # update the target network: new_target = tau * source + (1-tau) * target.
        source_module = self.network.model.modules_dict[module_name]
        target_module = self.network.model.modules_dict[f'target_{module_name}']
        for target_param, source_param in zip(target_module.parameters(), source_module.parameters()):
            target_param.data.copy_(self.config['tau'] * source_param.data +
                                    (1 - self.config['tau']) * target_param.data)

    def update(self, batch):
        """Update the agent and return self along with an info dictionary."""
        self.network.model.train()
        loss, info = self.total_loss(batch)
        self.network.optimizer.zero_grad()
        loss.backward()
        self.network.optimizer.step()
        self.network.step += 1

        self.target_update('critic')
        return self, info

    def sample_actions(self, observations, seed=None, temperature=1.0):
        if seed is None:
            seed = torch.Generator()
        # Determine noise shape.
        if self.config['ob_dims']:
            batch_shape = observations.shape[:-len(self.config['ob_dims'])]
        else:
            batch_shape = observations.shape
        noise_shape = batch_shape + (self.config['action_dim'],)
        noises = torch.randn(noise_shape, generator=seed)
        actions = self.network.model(**{
            'observations': observations,
            'actions': noises
        }, name='actor_onestep_flow')
        actions = torch.clamp(actions, -1, 1)
        return actions

    def compute_flow_actions(self, observations, noises):
        # If an encoder is defined, encode observations.
        if self.config['encoder'] is not None:
            observations = self.network.model(**{
                'observations': observations
            }, name='actor_bc_flow_encoder')
        actions = noises
        flow_steps = self.config['flow_steps']
        for i in range(flow_steps):
            t = torch.full((*observations.shape[:-1], 1), i / flow_steps, device=observations.device)
            vels = self.network.model(**{
                'observations': observations,
                'actions': actions,
                'times': t,
                'is_encoded': True
            }, name='actor_bc_flow')
            actions = actions + vels / flow_steps
        actions = torch.clamp(actions, -1, 1)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = torch.Generator()
        rng.manual_seed(seed)

        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]

        # encoders
        encoders = {}
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_flow'] = encoder_module()
            encoders['actor_onestep_flow'] = encoder_module()

        # networks
        critic_def = Value(
            input_dim=config['input_dim'],
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        actor_bc_flow_def = ActorVectorField(
            input_dim=int(np.prod(ob_dims)) + action_dim + 1,
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
        )
        actor_onestep_flow_def = ActorVectorField(
            input_dim=int(np.prod(ob_dims)) + action_dim,
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep_flow'),
        )

        network_info = {
            'critic': (critic_def, {'observations': ex_observations, 'actions': ex_actions}),
            'target_critic': (copy.deepcopy(critic_def), {'observations': ex_observations, 'actions': ex_actions}),
            'actor_bc_flow': (actor_bc_flow_def, {'observations': ex_observations, 'actions': ex_actions, 'times': ex_times}),
            'actor_onestep_flow': (actor_onestep_flow_def, {'observations': ex_observations, 'actions': ex_actions}),
        }
        if encoders.get('actor_bc_flow') is not None:
            network_info['actor_bc_flow_encoder'] = (encoders.get('actor_bc_flow'), {'observations': ex_observations})

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optim.Adam(network_def.parameters(), lr=config['lr'])
        for name, args in network_args.items():
            _ = network_def(**args, name=name)
        network_def.modules_dict['target_critic'].load_state_dict(
            network_def.modules_dict['critic'].state_dict()
        )

        network = TrainState.create(model=network_def, optimizer=network_tx)
        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        return cls(rng, network=network, config=config)

def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='fql_torch_sean',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions.
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension.
            input_dim=ml_collections.config_dict.placeholder(int),
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            actor_hidden_dims=[512, 512, 512, 512],  # Actor network hidden dimensions.
            value_hidden_dims=[512, 512, 512, 512],  # Value network hidden dimensions.
            layer_norm=True,  # Use layer normalization.
            actor_layer_norm=False,  # Use layer normalization for the actor.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            q_agg='mean',  # Aggregation method for target Q values.
            alpha=10.0,  # BC coefficient.
            flow_steps=10,  # Number of flow steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name.
        )
    )
    return config