import inspect as _inspect

async def _safe_await(maybe):
    """Await if awaitable; otherwise return value as-is."""
    if maybe is None:
        return None
    if _inspect.isawaitable(maybe):
        return await maybe
    return maybe
