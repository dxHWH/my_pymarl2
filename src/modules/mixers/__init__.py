from .qmix import QMixer
from .vdn import VDNMixer
from .qatten import QattenMixer
from .nqmix import NQMixer
from .attention_qmix import AttentionQMixer
from .dvd import DVDMixer
<<<<<<< HEAD
from .dvd_wm_mixer import DVDWMMixer      # <--- 新增
=======

>>>>>>> origin/main
REGISTRY = {
    "qmix": QMixer,
    "vdn": VDNMixer,
    "qatten": QattenMixer,
    "nqmix": NQMixer,
    "attention_qmix": AttentionQMixer,
    "dvd": DVDMixer,
<<<<<<< HEAD
    "dvd_wm":DVDWMMixer
}

=======
}
>>>>>>> origin/main
