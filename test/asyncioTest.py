# import asyncio
# import time
# async def async_function(task_name, seconds):
#     print(f"{task_name} started")
#     time.sleep(seconds)
#     # await asyncio.sleep(0)

#     print(f"{task_name} completed")

# def main():
#     for i in range (0,2):
#         asyncio.create_task(async_function("Task "+str(i), 3))
#         # asyncio.create_task(async_function("Task 2", 2))

#     print("Both tasks are running in the background")

#     # Continue doing other work here

#     # Do not wait for any tasks to complete

# async def intermediate():
#     main()

# asyncio.run(intermediate())


