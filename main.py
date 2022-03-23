import math
import time
import cv2
from mtcnn import MTCNN


# B is the point in the middle
def getAngle(coords_A, coords_B, coords_C):
    ang = math.degrees(math.atan2(coords_C[1] - coords_B[1], coords_C[0] - coords_B[0])
                       - math.atan2(coords_A[1] - coords_B[1], coords_A[0] - coords_B[0]))
    return ang + 360 if ang < 0 else ang


def detect_tilt(p_right_eye, p_nose, p_left_eye, p_right_mouth, p_left_mouth):
    upwards = 360 - getAngle(p_right_eye, p_nose, p_left_eye)
    downwards = getAngle(p_right_mouth, p_nose, p_left_mouth)
    left = 360 - getAngle(p_left_eye, p_nose, p_left_mouth)
    right = getAngle(p_right_eye, p_nose, p_right_mouth)

    return [upwards, downwards, left, right]


def main():
    detector = MTCNN()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)
    cap.set(cv2.CAP_PROP_FPS, 25)

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

                angles = detect_tilt(face['keypoints']['right_eye'], face['keypoints']['nose'],
                                     face['keypoints']['left_eye'],
                                     face['keypoints']['mouth_right'], face['keypoints']['mouth_left'])

                pitch = ((angles[0] - angles[1]) / 2) - 10
                yaw = (angles[3] - angles[2]) / 2

                cv2.putText(image, "pitch: %.2f" % pitch, (x + int(w / 2), y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                cv2.putText(image, "yaw: %.2f" % yaw, (x + w, y + int(h / 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                if angles[0] > 100:
                    cv2.putText(image, "Tilted upwards", (x + int(w / 2), y + h + int(h / 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                if angles[1] > 100:
                    cv2.putText(image, "Tilted downwards", (x + int(w / 2), y + h + int(h / 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                if angles[2] > 120:
                    cv2.putText(image, "Tilted right", (x + int(w / 2), y + h + int(h / 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                if angles[3] > 120:
                    cv2.putText(image, "Tilted left", (x + int(w / 2), y + h + int(h / 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                for label, keypoint in face['keypoints'].items():
                    cv2.circle(image, keypoint, 5, (c, c, 255), 2)

            cv2.imshow("Output", image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
