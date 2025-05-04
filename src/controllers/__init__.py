REGISTRY = {}

from .basic_controller import BasicMAC
from .original_basic_controller import BasicMAC as originalBasicMAC
REGISTRY["basic_mac"] = BasicMAC
REGISTRY["original_basic_mac"] = originalBasicMAC