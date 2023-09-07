import os

for key, value in os.environ.items():
    print('{}: {}'.format(key, value))

    