import os
from inference import get_model
import cv2
import utils
import numpy as np

# Set environment variable to suppress TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

model = get_model("argus-2/1", "FpwK2zTgVB5qgZJDUM7u")

os.system("cls")

counter = 1
cards = ["", "A", "2", "3", "4", "5", "6", "7", "8", "9", "J", "Q", "K"]

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    frame = cv2.resize(frame, (860, 720))

    if not ret:
        print("Unable to get image")
        break

    delaer = frame[0:300, :]
    player = frame[310:, :]
    key = cv2.waitKey(1)


    annotated_image = np.array(frame.copy())
    cv2.putText(
        annotated_image,
        f"Counter: {cards[int(counter % len(cards))]}",
        (100, 100),
        cv2.FONT_HERSHEY_COMPLEX,
        2,
        (255, 0, 255),
    )

    cv2.imshow("Camera", annotated_image)


    if key == ord("q"):
        break
    elif key == ord("s"):
        counter += 1

    # Dealer
    resultDealer = model.infer(frame, confidence=0.10)
    resDealer = utils.show_results(results=resultDealer, image=frame, card=cards[int(counter % len(cards))])
    cv2.imshow("Result", resDealer)

    # Player
    # resultPlayer = model.infer(player, confidence=0.10)
    # resPlayer = utils.show_results(results=resultPlayer, image=player)
    # cv2.imshow("Player", resPlayer)
