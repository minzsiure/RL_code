from agents.nfql import NFQLAgent
from agents.fql_torch import FQLAgent_Torch
# from agents.fql_torch_sean import FQLAgent_Torch

agents = dict(
    fql_torch_sean=FQLAgent_Torch,
    nfql=NFQLAgent,
)
