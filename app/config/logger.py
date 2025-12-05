import logging
import sys
import os
from logging.handlers import RotatingFileHandler
from app.config.config import config


class LoggerSetup:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LoggerSetup, cls).__new__(cls)
            cls._instance._initialize_logger()
        return cls._instance

    def _initialize_logger(self):
        # 1. Get Config from Environment Variables (with defaults)
        log_level_str = config.log_level
        log_file = config.log_file_path
        max_bytes = config.log_max_bytes
        backup_count = config.log_backup_count

        # Convert string level to logging constant
        log_level = getattr(logging, log_level_str, logging.INFO)

        # 2. Create the Logger
        self.logger = logging.getLogger("HR Agent System")
        self.logger.setLevel(log_level)

        # Prevent adding duplicate handlers if script is re-imported
        if self.logger.hasHandlers():
            return

        # 3. Formatter (Best practice: include time, level, file, line number)
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(levelname)-8s - %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # 4. Handler: Stream (Stdout/Console)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # 5. Handler: File with Rotation
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=max_bytes, 
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def get_logger(self):
        return self.logger

# Create a singleton instance
logger = LoggerSetup().get_logger()