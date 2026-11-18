import sys
import inspect

# 1. Set your global debug level (0=Off, 1=Info, 2=Debug, 3=Verbose)
LOG_OFF = 0
LOG_ERROR = LOG_OFF+1
LOG_WARN = LOG_OFF+2
LOG_INFO = LOG_OFF+3
LOG_DEBUG = LOG_OFF+4
LOG_VERBOSE = LOG_OFF+5

DEBUG_LEVEL = LOG_OFF

# --- Colors & Symbols ---
COLORS = {
    LOG_ERROR: "\033[91m",  # Red
    LOG_WARN:  "\033[93m",  # Yellow
    LOG_INFO:  "\033[97m",  # Green
    LOG_DEBUG: "\033[94m",  # Blue
    LOG_VERBOSE: "\033[96m", # Cyan
}
RESET = "\033[0m"

SYMBOLS = {
    LOG_ERROR: "❌",
    LOG_WARN:  "⚠️",
    LOG_INFO:  "ℹ️",
    LOG_DEBUG: "🔍",
    LOG_VERBOSE: "📝",
}

def set_logger_level(level):
    global DEBUG_LEVEL
    DEBUG_LEVEL = level

def LOG(level, message, module=None):
    """
    level: 1=Info, 2=Debug, 3=Verbose
    module: Optional. If None, auto-detects from caller.
    """
    if level <= DEBUG_LEVEL:
        if module is None:
            # Auto-detect module name from the caller's frame (2 levels up: log -> caller)
            frame = inspect.currentframe().f_back.f_back
            module_name = frame.f_globals.get('__name__', 'Unknown').split('.')[-1]
            func_name = frame.f_code.co_name[:15]
            line_no = frame.f_lineno
        
        color = COLORS.get(level, RESET)
        symbol = SYMBOLS.get(level, "?")
    
        # Format: [Symbol] Module | Message
        print(f"{color}{symbol}  {func_name:<15}:{line_no:<4} | {message}", file=sys.stderr)

def LOG_ERR(message, module=None):
    LOG(LOG_ERR, message, module)

def LOG_WRN(message, module=None):
    LOG(LOG_WARN, message, module)

def LOG_DBG(message, module=None):
    LOG(LOG_DEBUG, message, module)

def LOG_INF(message, module=None):
    LOG(LOG_INFO, message, module)

def LOG_VRB(message, module=None):
    LOG(LOG_VERBOSE, message, module)

