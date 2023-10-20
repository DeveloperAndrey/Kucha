import os
from time import sleep
from random import randint
import subprocess
import shutil
from json import loads, dumps

from django.core.mail import send_mail
from django.conf import settings
from celery import shared_task
import redis


@shared_task()
def make_nn_task(redis_key, file):
    redis_instance = redis.StrictRedis(host=settings.REDIS_HOST,
                                       port=settings.REDIS_PORT)

    redis_instance.set(f"status_{redis_key}", 'run')

    for i in range(200):
        sleep(5)

        work_list = loads(redis_instance.get(f"work_{redis_key}"))
        work_list.append({"time": i, "v": i**2})
        redis_instance.set(f"work_{redis_key}", dumps(work_list))
    redis_instance.set(f"status_{redis_key}", 'done')
