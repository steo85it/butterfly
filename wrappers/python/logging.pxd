cdef extern from "bf/logging.h":
    ctypedef enum BfLogLevel:
        BF_LOG_LEVEL_TODO
        BF_LOG_LEVEL_DEBUG
        BF_LOG_LEVEL_INFO
        BF_LOG_LEVEL_WARN
        BF_LOG_LEVEL_ERROR

    void bfSetLogLevel(BfLogLevel logLevel)
