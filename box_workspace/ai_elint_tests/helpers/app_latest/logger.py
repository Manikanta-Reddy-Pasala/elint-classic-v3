import sys
import logging

# logging.basicConfig(
#     # level=logging.INFO,
#     # format="%(levelname)s: %(name)s.%(funcName)s:%(lineno)d - %(message)s",
#     format="%(levelname)1s: %(funcName)10s:%(lineno)d - %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S")

logging.VERBOSE = 5
logging.addLevelName(logging.VERBOSE, "VERBOSE")

logging.MEDIUM_PRIO_INFO = 25
logging.addLevelName(logging.MEDIUM_PRIO_INFO, "MED_PRIO")

logging.HIGH_PRIO_INFO = 30  # Note: overlaps standard CRITICAL (50) in magnitude, but we treat it as a distinct logical level
logging.addLevelName(logging.HIGH_PRIO_INFO, "HIGH_PRIO")

# Add convenience methods to the Logger class
def verbose(self, message, *args, **kwargs):
    if self.isEnabledFor(logging.VERBOSE):
        self.log(logging.VERBOSE, message, *args, **kwargs)

def medium_prio(self, message, *args, **kwargs):
    if self.isEnabledFor(logging.MEDIUM_PRIO_INFO):
        self.log(logging.MEDIUM_PRIO_INFO, message, *args, **kwargs)

def high_prio(self, message, *args, **kwargs):
    if self.isEnabledFor(logging.HIGH_PRIO_INFO):
        self.log(logging.HIGH_PRIO_INFO, message, *args, **kwargs)

logging.Logger.verbose = verbose
logging.Logger.medium_prio = medium_prio
logging.Logger.high_prio = high_prio

# --- 2. DEFINE COLORS & ICONS ---

# ANSI Color Codes
COLOR_RESET = "\033[0m"
COLOR_BLACK_BOLD = "\033[1;30m"
COLOR_RED_BOLD = "\033[1;91m"
COLOR_GREEN_BOLD = "\033[1;92m"
COLOR_YELLOW_BOLD = "\033[1;93m"
COLOR_YELLOW = "\033[93m"
COLOR_CYAN_BOLD = "\033[1;96m"       # Medium Priority
COLOR_MAGENTA_BOLD = "\033[1;95m"    # High Priority

LEVEL_CONFIG = {
    logging.VERBOSE:      ("🔎", COLOR_BLACK_BOLD, "VERBOSE"),
    logging.DEBUG:      ("🔍", COLOR_RESET, "DEBUG"),
    logging.INFO:       ("ℹ️", COLOR_CYAN_BOLD,   "INFO"),
    logging.WARNING:    ("⚠️", COLOR_YELLOW_BOLD, "WARN"),
    logging.ERROR:      ("❌", COLOR_RED_BOLD,    "ERROR"),
    logging.CRITICAL:   ("💥", COLOR_RED_BOLD,    "CRIT"),
    logging.MEDIUM_PRIO_INFO:   ("⚡", COLOR_YELLOW, "MEDIUM"),
    logging.HIGH_PRIO_INFO:     ("🚨", COLOR_MAGENTA_BOLD, "HIGH"),
}

class CustomFormatter(logging.Formatter):
    def __init__(self):
        # Empty fmt disables automatic level name insertion, allowing manual control
        super().__init__('')
    def format(self, record):
        emoji, color, label = LEVEL_CONFIG.get(record.levelno,  ("❓", COLOR_RESET, "?"))
                
        # Format module name
        if record.name == '__main__':
            module_name = '**main**'
        else:
            module_name = f'**{record.name}**'
        
        # Format function name (fixed width)
        func_name = record.funcName
        if func_name == '<module>':
            func_name = f"{'MAIN':<15}"
        else:
            func_name = func_name[:15].ljust(15)
            
        # Get the message
        message = record.getMessage()
        
        # Add exception traceback if present
        if record.exc_info and record.exc_info[0] is not None:
            if record.exc_text:
                message = message + '\n' + record.exc_text.strip()
            else:
                import traceback
                message = message + '\n' + ''.join(traceback.format_exception(*record.exc_info)).strip()
        
        # Construct message
        return f"{color}{emoji[:2]} : {func_name}:{record.lineno} | {message}{COLOR_RESET}"
        # ---------------------------

def setup_logging(level=logging.ERROR):
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    if root_logger.handlers:
        root_logger.handlers.clear()

    # Clear any existing handlers (prevents duplicates if called twice)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add our custom handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(CustomFormatter())
    root_logger.addHandler(console_handler)

