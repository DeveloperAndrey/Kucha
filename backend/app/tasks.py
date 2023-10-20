import os
from time import sleep
from random import randint
import subprocess
import shutil
from json import loads

from django.core.mail import send_mail
from django.conf import settings
from celery import shared_task
import redis
import cv2
import mediapipe


from code import use_nn
from integr_second_nn import pred

@shared_task()
def make_nn_task(redis_key, file):
    redis_instance = redis.StrictRedis(host=settings.REDIS_HOST,
                                       port=settings.REDIS_PORT)

    redis_instance.set(f"status_{redis_key}", 'run')



    drawingModule = mediapipe.solutions.drawing_utils
    handsModule = mediapipe.solutions.hands

    capture = cv2.VideoCapture('../uploads/' + file)

    frameNr = -1

    with handsModule.Hands() as hands:

        while (True):

            success, frame = capture.read()

            if not success:
                break

            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks != None:
                for handLandmarks in results.multi_hand_landmarks:
                    drawingModule.draw_landmarks(frame, handLandmarks, handsModule.HAND_CONNECTIONS)

            frameNr = frameNr + 1

            if frameNr % 10 != 0:
                continue

            if use_nn(file) == True:
                scale = pred

    capture.release()




    work_list = loads(redis_instance.get(f"work_{redis_key}"))
    work_list.append({'time': 1, 'v': 45})
    redis_instance.set(f"work_{redis_key}", str(work_list))
    redis_instance.set(f"status_{redis_key}", 'done')
