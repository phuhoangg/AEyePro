from concurrent.futures import ThreadPoolExecutor

class ExecutorService:
    def __init__(self, max_workers: int = 4):
        self._executor = ThreadPoolExecutor(max_workers=max_workers,
                                            thread_name_prefix="AEye")
    def submit(self, fn, *a, **kw):
        return self._executor.submit(fn, *a, **kw)
    def shutdown(self):
        self._executor.shutdown(wait=True)