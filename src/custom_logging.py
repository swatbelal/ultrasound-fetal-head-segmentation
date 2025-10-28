import logging
import config as cfg

log_file = cfg.LOG_FILE_NAME
log_level = cfg.LOG_LEVEL

file_handler = logging.FileHandler(log_file, mode="w", delay=True, encoding='utf-8')

if log_level == "debug":
    actual_log_level = logging.DEBUG
elif log_level == "info":
    actual_log_level = logging.INFO
elif log_level == "warning":
    actual_log_level = logging.WARNING
elif log_level == "error":
    actual_log_level = logging.ERROR
elif log_level == "critical":
    actual_log_level = logging.CRITICAL
else:
    actual_log_level = logging.DEBUG

log_format = "%(asctime)s - %(levelname)s - %(name)s:%(module)s - %(threadName)s - %(message)s"  # %(module)s ` %(name)s

logger = logging
logging.basicConfig(
    level=actual_log_level,
    format=log_format,
    handlers=[
        file_handler,
        logging.StreamHandler()
    ]
)


# Set up a single logger and handler our modules
combined_log_file = "dump.log"

# Configure combined logger
combined_logger = logging.getLogger("combined_logger")
combined_handler = logging.FileHandler(combined_log_file, mode="w", delay=True, encoding="utf-8")
combined_handler.setFormatter(logging.Formatter(log_format))
combined_logger.addHandler(combined_handler)
combined_logger.setLevel(actual_log_level)
combined_logger.propagate = False  # Prevent logs from being propagated to the parent logger

# Attach the combined logger to our modules
modules_to_combine = [
    "tensorflow",
    "h5py._conv",
    "PIL.TiffImagePlugin",
    "PIL.PngImagePlugin",
]

for module_name in modules_to_combine:
    module_logger = logging.getLogger(module_name)
    module_logger.addHandler(combined_handler)
    module_logger.setLevel(actual_log_level)
    module_logger.propagate = False
