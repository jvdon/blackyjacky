import cv2
import numpy as np


WIDTH = 1920
HEIGHT = 1080
cards = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "J", "Q", "K"]


def show_results(results, image: cv2.Mat, card: int):
    annotated_image = np.array(image.copy())
    img_height, img_width, _ = image.shape

    SCALE_X = WIDTH / img_width
    SCALE_Y = HEIGHT / img_height

    for result in results:
        for prediction in result.predictions:
            if prediction.class_name in ["J", "Q", "K", "A"]:
                text = prediction.class_id
            else:
                text = int(prediction.class_name) - 1

            if str(text) == str(card):
                cv2.putText(
                    annotated_image,
                    "MATCH",
                    # f"{str(card)} {text} {str(str(text) == str(card))}",
                    (200, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (255, 0, 255),
                )
            else:
                cv2.putText(
                    annotated_image,
                    "NOT MATCH",
                    # f"{str(card)} {text} {str(str(text) == str(card))}",
                    (200, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (255, 0, 255),
                )

            x = int(prediction.x - prediction.width / 2)
            y = int(prediction.y - prediction.height / 2)
            x2 = int(prediction.x + prediction.width / 2)
            y2 = int(prediction.y + prediction.height / 2)

            xS = int(prediction.x * SCALE_X)
            yS = int(prediction.y * SCALE_Y)

            # pydirectinput.click(xS, yS, interval=0.5)

            cv2.rectangle(annotated_image, (x, y), (x2, y2), (0, 255, 255), 2)
            if prediction.class_name in ["J", "Q", "K", "A"]:
                text = prediction.class_id
            else:
                text = int(prediction.class_name) - 1

            cv2.putText(
                annotated_image,
                str(text),
                (x, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
            )

            cv2.circle(
                annotated_image,
                (xS, yS),
                2,
                (255, 255, 255),
                -1,
            )

            cv2.circle(
                annotated_image,
                (int(prediction.x), int(prediction.y)),
                2,
                (0, 255, 255),
                -1,
            )
    return annotated_image
