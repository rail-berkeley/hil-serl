import numpy as np
import threading
import cv2

class ImageDisplayer(threading.Thread):
    def __init__(self, queue, name):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True  # make this a daemon thread
        self.name = name

    def run(self):
        while True:
            img_array = self.queue.get()  # retrieve an image from the queue
            if img_array is None:  # None is our signal to exit
                break

            assert isinstance(img_array, dict)
            frame = np.concatenate(
                [cv2.resize(v, (1400, 400)) for k, v in img_array.items() if "full" not in k], axis=1
            )

            cv2.imshow(self.name, frame)
            cv2.waitKey(1)


def q_image(q_value, q_value_grasp: np.ndarray, q_value_grasp_index: int, intervene: bool):
    q_image = np.zeros((400, 1400, 3), dtype=np.uint8)
    q_image = cv2.putText(
        img = q_image,
        text = f"qv={round(float(q_value), 4)}",
        org = (10, 40),
        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = 1,
        color = (255, 255, 255),
        thickness = 2,
        lineType = cv2.LINE_AA,
    )
    q_image = cv2.putText(
        img = q_image,
        text = f"qv_grasp={[round(float(q), 4) for q in q_value_grasp]} ({['open', 'stay', 'close'][q_value_grasp_index]})",
        org = (10, 140),
        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = 1,
        color = (255, 255, 255),
        thickness = 2,
        lineType = cv2.LINE_AA,
    )
    if intervene:
        q_image = cv2.putText(
            img = q_image,
            text = f"intervening",
            org = (10, 240),
            fontFace = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 1,
            color = (255, 255, 255),
            thickness = 2,
            lineType = cv2.LINE_AA,
        )
    return q_image


def ema(series, alpha=0.5):
    """
    Exponential moving average
    :param series: the input series
    :param alpha: the smoothing factor
    :return: the smoothed series
    """
    smoothed = np.zeros_like(series, dtype=float)
    smoothed[0] = series[0]
    for i in range(1, len(series)):
        smoothed[i] = alpha * series[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed
