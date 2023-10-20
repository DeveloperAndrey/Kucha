from time import sleep
from integr_second_nn import neural_scale
REDIS_HOST = 'redis'
REDIS_PORT = 6379



def neural_network():
    neural_scale()
    for i in range(600):
        print(i)
        sleep(1)

    return 'Done'