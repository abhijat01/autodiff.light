import logging

log = logging.getLogger("ad.core")  # ad stands for auto diff
log.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("autodiff.log")
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_formatter = logging.Formatter('%(message)s')
file_handler.setFormatter(file_formatter)
console_handler.setFormatter(console_formatter)
handlers = {'file': file_handler, 'console': console_handler}
for handler in handlers.values():
    log.addHandler(handler)


def log_level_info():
    log_debug("Setting level for all handlers to INFO")
    for h in handlers.values():
        h.setLevel(logging.INFO)


def log_level_debug():
    for h in handlers.values():
        h.setLevel(logging.DEBUG)


def log_debug(message):
    log.debug(message)


def log_info(message):
    log.info(message)