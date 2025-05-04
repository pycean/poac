REGISTRY = {}

from .rnn_agent import RNNAgent
from .original_rnn_agent import RNNAgent as originalRNNAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["original_rnn"] = originalRNNAgent