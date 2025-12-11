from .qmix import QMixer
from .vdn import VDNMixer
from .qatten import QattenMixer
from .nqmix import NQMixer
from .attention_qmix import AttentionQMixer
from .dvd import DVDMixer

REGISTRY = {
    "qmix": QMixer,
    "vdn": VDNMixer,
    "qatten": QattenMixer,
    "nqmix": NQMixer,
    "attention_qmix": AttentionQMixer,
    "dvd": DVDMixer,
}
