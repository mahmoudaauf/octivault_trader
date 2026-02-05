# core/errors.py

class BinanceAPIException(Exception):
    def __init__(self, message, code=None):
        self.message = message
        self.code = code
        super().__init__(f"API Error: {message} (Code: {code})")

class NetworkException(Exception):
    def __init__(self, message="Network communication failed"):
        super().__init__(message)
