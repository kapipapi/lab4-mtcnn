import cv2
import time
from mtcnn import MTCNN


def main():
    detector = MTCNN()

    cap = cv2.VideoCapture(0)
    while True:
        ret, image = cap.read()
        if not ret:
            raise IOError("webcam failure")
        if image is not None:
            t0 = time.time()
            result = detector.detect_faces(image)
            t = time.time() - t0
            print('Time: ', t)

            for face in result:
                c = int(255 * (1 - face['confidence']))
                x, y, w, h = face['box']
                cv2.rectangle(image, (x, y), (x + w, y + h), (c, c, 255), 2)

                for label, keypoint in face['keypoints'].items():
                    cv2.circle(image, keypoint, 5, (c, c, 255), 2)

            cv2.imshow("Output", image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
