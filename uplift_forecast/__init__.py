__all__ = ['UpliftForecast', 'UpliftModel', 'BaseMetaUpliftModel', 'BaseNeuralUpliftModel', 'RERUM']

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version('uplift-forecast')
except PackageNotFoundError:
    __version__ = '0.1.0'

from .common import BaseMetaUpliftModel, BaseNeuralUpliftModel, UpliftModel
from .core import UpliftForecast
from .frameworks import RERUM
