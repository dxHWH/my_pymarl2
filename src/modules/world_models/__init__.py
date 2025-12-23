from .vae_rnn import VAERNNWorldModel
from .rssm_model import RSSMWorldModel  

REGISTRY = {}

REGISTRY["vae_rnn"] = VAERNNWorldModel
REGISTRY["rssm"] = RSSMWorldModel