from enum import StrEnum


class AppErrorCode(StrEnum):

    INVALID_ID = "INVALID_ID"
    INVALID_INPUT = "INVALID_INPUT"
    NOT_FOUND = "NOT_FOUND"
    AUTH_FAILED = "AUTH_FAILED"
    TIMEOUT = "TIMEOUT"
    INTERNAL = "INTERNAL"
