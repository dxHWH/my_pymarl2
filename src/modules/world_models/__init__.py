from .vae_rnn import VAERNNWorldModel
from .rssm_model import RSSMWorldModel 
from .rssm_model_origin import RSSMWorldModelorigin 

REGISTRY = {}

REGISTRY["vae_rnn"] = VAERNNWorldModel
REGISTRY["rssm"] = RSSMWorldModel
REGISTRY["rssm_origin"] = RSSMWorldModelorigin