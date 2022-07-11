"""
Created on Thu Jun 23 09:40:24 2022

@author: Stefano Talamona
"""

from telegram.ext import *
from io import BytesIO
import cv2 as cv
import numpy as np
import time
from iqa import isImageOk, isVideoOk

import sys
sys.path.insert(0, 'D:/UNIVERSITA/Magistrale/SecondoAnno/VisualInformationProcessingAndManagement/ProgettoVISUAL/code/darknet_scripts')
from darknet_helper_functions import detection_image, detection_video, load_networks


TOKEN = 'THIS_IS_A_PERSONAL_TOKEN'
CHOSEN_MODEL = 'yolov4'
CONFIDENCE = True

print("\n---LOADING THE MODELS---\n")
YOLOV4_NET, YOLOV4TINY_NET, CLASS_NAMES, CLASS_COLORS = load_networks()
print("\n---MODELS LOADED---\n")



def start(update, context):
    start_message = "Bot Telegram per il progetto d'esame del corso di Visual Information Processing and Management 🎓\n"
    start_message = start_message + "Appello dell'11 Luglio a.a. 2021/2022 \n"
    start_message = start_message + "Pothole Detection with Darknet framework and YOLOV4"
    update.message.reply_text(start_message)
   

def help(update, context):
    help_message = "/start - avvia la conversazione \n"
    help_message = help_message + "/help - visualizza questo messaggio \n"
    help_message = help_message + "/yolov4 - utilizza YOLOV4 per la detection (default) \n"
    help_message = help_message + "/yolov4tiny - utilizza YOLOV4-tiny per la detection \n"
    help_message = help_message + "/bboxes_only - mostra solo le bounding box nei risultati della detection \n"
    help_message = help_message + "/show_confidence - mostra anche le percentuali di confidenza nei risultati della detection (default)\n"
    help_message = help_message + "\nCarica una immagine o un video per eseguire la detection delle eventuali buche 🔎🕳"
    update.message.reply_text(help_message)
    

def handle_message(update, context):
    update.message.reply_text('Digita "/help" per visualizzare le opzioni disponibili, o carica direttamente una immagine o un video! 📸')



def handle_photo(update, context):
    # download image
    file = context.bot.get_file(update.message.photo[-1].file_id)
    f = BytesIO(file.download_as_bytearray())
    file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
    img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    
    # check image quality
    if not isImageOk(img):
        update.message.reply_text("❗ Attenzione ❗\nLa qualità dell'immagine non è sufficiente per eseguire una detection accurata.")
        return 0
    
    # check image resolution
    dims = img.shape[:-1]
    if (max(dims) / min(dims)) >= 1.8:
        update.message.reply_text("❗ Attenzione ❗\nLe dimensioni dell'immagine non sono adatte ad eseguire una detection accurata.")
        return 0

    # potholes detection
    if CHOSEN_MODEL == 'yolov4':
        result, n_potholes, detection_time = detection_image(img, YOLOV4_NET, CLASS_NAMES, CLASS_COLORS, CONFIDENCE, font_size=1.25, bbox_thickness=4)
    else:
        result, n_potholes, detection_time = detection_image(img, YOLOV4TINY_NET, CLASS_NAMES, CLASS_COLORS, CONFIDENCE, font_size=1.25, bbox_thickness=4)
    
    # send text result
    if n_potholes == 0:
        potholes_message = "Detection conclusa! \nNon sembrano esserci buche nell'immagine in input 👍"
        update.message.reply_text(potholes_message)
        return 0
    elif n_potholes == 1:
        potholes_message = "Detection conclusa! \nAttenzione, sembra esserci una buca nell'immagine in input! ⚠"
    else:
        potholes_message = f"Detection conclusa! \nAttenzione, ci sono {n_potholes} buche nell'immagine in input! ⚠"
    potholes_message = potholes_message + f"\nIl processo di pothole detection ha impiegato {round(detection_time, 2)} secondi ({CHOSEN_MODEL}) 🕰"
    update.message.reply_text(potholes_message)
    
    # send image result
    img_path = './result.jpg'
    cv.imwrite(img_path, result)
    context.bot.sendPhoto(chat_id=update.effective_chat.id, photo=open(img_path, 'rb'))
    
    # print current status of the bot
    print("\n---DETECTION COMPLETED---\n")
    print("---BOT IS LISTENING---\n")



def handle_video(update, context):
    update.message.reply_text("Questa funzionalità non è al momento disponibile 💔")
    return 0
    
    # download video
    update.message.reply_text("Downloading del video in corso ⏳...")
    video = context.bot.get_file(update.message.video[-1].file_id)
    update.message.reply_text("Video ricevuto, detection in corso 🔎...")

    # check video quality
    if not isVideoOk(video):
        update.message.reply_text("❗ Attenzione ❗\nLa qualità del video non è sufficiente per eseguire una detection accurata.")
        return 0

    # potholes detection
    if CHOSEN_MODEL == 'yolov4':
        frames_list, detections_per_frame, fps = detection_video(video, YOLOV4_NET, CLASS_NAMES, CLASS_COLORS, CONFIDENCE)
    else:
        frames_list, detections_per_frame, fps = detection_video(video, YOLOV4TINY_NET, CLASS_NAMES, CLASS_COLORS, CONFIDENCE)
    
    # compose the video form the sequence of frames
    size = np.shape(frames_list[0])[:-1]
    video_path = './result.mp4'
    output_video = cv.VideoWriter(video_path, cv.VideoWriter_fourcc(*'mp4v'), 30, (size[1], size[0]))
    for img in frames_list:
        output_video.write(img)
    output_video.release()

    # send text result
    if detections_per_frame > 0:
        potholes_message = f"Detection conclusa! Attenzione, sono state individuate in media {detections_per_frame} buche per frame ⚠"
        potholes_message = potholes_message + f"\nIl processo di detection è avvenuto a una velocità di elaborazione pari a {fps}fps ({CHOSEN_MODEL}) 📈"
        potholes_message = potholes_message + "\nInvio del video risultato in corso 📩..."
        update.message.reply_text(potholes_message)
        # send video result
        context.bot.sendVideo(chat_id=update.effective_chat.id, video=open(video_path, 'rb'))
    else:
        potholes_message = "Detection conclusa! Non sembrano esserci buche nel video in input 👍"
    update.message.reply_text(potholes_message)

    # print current status of the bot
    print("\n---DETECTION COMPLETED---\n")
    print("---BOT IS LISTENING---\n")



def set_yolov4(update, context):
    global CHOSEN_MODEL
    CHOSEN_MODEL = 'yolov4'
    update.message.reply_text("Hai selezionato l'architettura YOLOV4 per la detection delle buche ✅")


def set_yolov4tiny(update, context):
    global CHOSEN_MODEL
    CHOSEN_MODEL = 'yolov4-tiny'
    update.message.reply_text("Hai selezionato l'architettura YOLOV4-tiny per la detection delle buche ✅")


def set_bboxes(update, context):
    global CONFIDENCE
    CONFIDENCE = False
    update.message.reply_text("Verranno mostrate solo le bounding box nei risultati della detection ✅")


def set_confidence(update, context):
    global CONFIDENCE
    CONFIDENCE = True
    update.message.reply_text("Verranno mostrate anche le percentuali di confidenza del modello nei risultati della detection ✅")



updater = Updater(TOKEN, use_context=True)
dp = updater.dispatcher

dp.add_handler(CommandHandler("start", start))
dp.add_handler(CommandHandler("help", help))
dp.add_handler(CommandHandler("yolov4", set_yolov4))
dp.add_handler(CommandHandler("yolov4tiny", set_yolov4tiny))
dp.add_handler(CommandHandler("bboxes_only", set_bboxes))
dp.add_handler(CommandHandler("show_confidence", set_confidence))
dp.add_handler(MessageHandler(Filters.text, handle_message))
dp.add_handler(MessageHandler(Filters.photo, handle_photo))
dp.add_handler(MessageHandler(Filters.video, handle_video))


print("\n---BOT IS LISTENING---\n")
updater.start_polling()
updater.idle()
print("\n---BOT DISCONNECTED---\n")
