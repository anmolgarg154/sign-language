from flask import Flask, render_template, request, jsonify, send_from_directory # type: ignore
import cv2
import numpy as np
import base64
from cvzone.HandTrackingModule import HandDetector # type: ignore
from cvzone.ClassificationModule import Classifier # type: ignore
import math
import time 

# import threading
# from playsound import playsound



app = Flask(__name__)

detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

labels = ["hello welcome to UPITS 2024","what is your name","how are you","where are you from","have a great day"]

word = ""
update_interval = 1  
last_update_time = time.time()  

# def play_audio(word):
#     audio_thread = threading.Thread(target=lambda: playsound(f'audios/{word}.wav')).start()



def update_word(letter):
    global word
    if letter == word:
        pass
    else:
        word = letter
        # play_audio(word)
    print("\n\n\n\n\n\n\n\nWord : ", word)

def process_frame(frame_data):
    try:
        img_bytes = base64.b64decode(frame_data.split(',')[1])
        img_np = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        global word
        global last_update_time  

        offset = 20
        imgSize = 300

        imgOutput = img.copy()
        hands, _ = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            imgCropShape = imgCrop.shape
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                if hCal > 230:
                    print("Invalid size to resize image.")
                else:
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)

            current_time = time.time()
            if current_time - last_update_time >= update_interval:
                letter = labels[index]
                update_word(letter)
                last_update_time = current_time

            print("word: ", word, "labels[index] ", labels[index])
            return word
        
        else:
            word = ""

    except Exception as e:
        print(f"The exception {e} occurred! ")
        return 'Error occurred'

@app.route('/')
def index():
    return render_template('index2.html', predicted_text='')

@app.route('/process_frame', methods=['POST'])
def process_frame_route():
    try:
        data = request.get_json()
        frame_data = data['frame']
        predicted_text = process_frame(frame_data)
        
        print(f"predicted text {predicted_text} , word {word} ")

        return jsonify({'predicted_text': word})
    except Exception as e:
        print(f"The exception {e} occurred! ")
        return jsonify({'predicted_text': 'Error occurred'})



@app.route('/audios/<path:filename>', methods=['GET'])
def send_audio(filename):
    return send_from_directory('audios/new', filename)



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

