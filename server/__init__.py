import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def getLogger(name: str) -> logging.Logger:
    return logging.getLogger(name=name)
