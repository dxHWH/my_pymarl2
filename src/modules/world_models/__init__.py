from .vae_rnn import VAERNNWorldModel
from .rssm_model import RSSMWorldModel 
from .rssm_model_origin import RSSMWorldModelorigin 
from .vae_rnn_attention import VAERNNAttentionWorldModel

REGISTRY = {}

REGISTRY["vae_rnn"] = VAERNNWorldModel
REGISTRY["rssm"] = RSSMWorldModel
REGISTRY["rssm_origin"] = RSSMWorldModelorigin
REGISTRY["vae_rnn_atten"] =  VAERNNAttentionWorldModel