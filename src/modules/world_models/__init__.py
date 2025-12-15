from .vae_rnn import VAERNNWorldModel
# from .rssm import RSSMWorldModel  <-- 未来这一行解开注释即可

REGISTRY = {}

REGISTRY["vae_rnn"] = VAERNNWorldModel
# REGISTRY["rssm"] = RSSMWorldModel