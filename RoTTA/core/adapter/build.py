from RoTTA.core.adapter.zeroshot_multiLabels import ZeroshotMultiLabels
from RoTTA.core.adapter.tent_multilabel import TentMultiLabel
from RoTTA.core.adapter.rotta_multilabels import RoTTA_MultiLabels
from .base_adapter import BaseAdapter
from .rotta import RoTTA

def build_adapter(cfg) -> type(BaseAdapter):
    adapterName = cfg.ADAPTER.NAME
    print(f'Using adapter: {adapterName}')
    if adapterName == "rotta":
        return RoTTA
    elif adapterName == "rotta_multilabels":
        return RoTTA_MultiLabels
    elif adapterName == 'tent':
        return TentMultiLabel
    elif adapterName == "zeroshot":
        return ZeroshotMultiLabels
    else:
        raise NotImplementedError("Implement your own adapter")

