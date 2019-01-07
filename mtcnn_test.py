import cv2 as cv
from mtcnn.mtcnn import MTCNN


def worker(img):
    # Resize image to 320x240
    img = cv.resize(img, (320, 240), 0, 0)

    # Detect faces position
    result = detector.detect_faces(img)

    # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
    if (result):
        bounding_box = result[0]['box']
        keypoints = result[0]['keypoints']

        # Bounding face rect
        cv.rectangle(img,
                     (bounding_box[0], bounding_box[1]),
                     (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                     (0, 155, 255),
                     2)

        # Bounding face components
        cv.circle(img, (keypoints['left_eye']), 2, (0, 155, 255), 2)
        cv.circle(img, (keypoints['right_eye']), 2, (0, 155, 255), 2)
        cv.circle(img, (keypoints['nose']), 2, (0, 155, 255), 2)
        cv.circle(img, (keypoints['mouth_left']), 2, (0, 155, 255), 2)
        cv.circle(img, (keypoints['mouth_right']), 2, (0, 155, 255), 2)

    # Show result
    cv.imshow("Face", img)


if __name__ == '__main__':
    detector = MTCNN()
    cam = cv.VideoCapture(0)

    while (True):

        # Read image from cam in BGR
        _, image = cam.read()

        # Convert type image BGR to RGB
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        worker(image)

        # Exit on pressing Esc
        if cv.waitKey(1) == 27:
            break

    cv.destroyAllWindows()
