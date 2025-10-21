# # web_app/server/jobs/limiter.py
# import asyncio
# from contextlib import asynccontextmanager
# from utils import Throttle, Debounce

# @asynccontextmanager
# async def task_lock(lock: asyncio.Lock):
#     await lock.acquire()
#     try:
#         yield
#     finally:
#         lock.release()

# class JobLimiter:
#     """Guards long tasks from running concurrently."""
#     def __init__(self):
#         self._locks = {}

#     def lock_for(self, name: str) -> asyncio.Lock:
#         return self._locks.setdefault(name, asyncio.Lock())

# job_limiter = JobLimiter()
