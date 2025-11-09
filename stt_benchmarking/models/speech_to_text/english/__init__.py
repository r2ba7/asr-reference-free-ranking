from stt_benchmarking.utils import logger
from stt_benchmarking.utils.english_normalizer import normalizer

LOGGER = logger.Logger.get_logger(module_name=__name__).logger
NORMALIZER_OBJ = normalizer.EnglishTextNormalizer()

__all__ = [LOGGER, NORMALIZER_OBJ]