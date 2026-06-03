__all__ = ['RERUM', 'BaseMetaUpliftModel', 'BaseNeuralUpliftModel', 'UpliftForecast', 'UpliftModel']

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version('uplift-forecast')
except PackageNotFoundError:
    __version__ = '0.1.0'

from .common import BaseMetaUpliftModel, BaseNeuralUpliftModel, UpliftModel
from .core import UpliftForecast
from .frameworks import RERUM
