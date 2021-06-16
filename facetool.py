# FaceTool
# Source partially based on http://datahacker.rs/010-how-to-align-faces-with-opencv-in-python/

import cv2
import numpy as np
import os
from os import path
from os import listdir
from os.path import isfile, join
import mediapipe as mp


class Face:
    area = 0
    x = 0
    y = 0
    width = 0
    height = 0
    image = None
    left_eye = (0, 0)
    right_eye = (0, 0)


class FaceTool:
    face_cascade = None
    eye_cascade = None
    debug = False

    face_detection = None
    face_mesh = None

    def __init__(self, debug=False) -> None:
        self.debug = debug
        mp_face_detection = mp.solutions.face_detection
        self.face_detection = mp_face_detection.FaceDetection(
            min_detection_confidence=0.5)

    def imcrop(self, img, bbox):
        x2, y2, x1, y1 = np.array(bbox, dtype='int')
        if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
            img, x1, x2, y1, y2 = self.pad_img_to_fit_bbox(img, x1, x2, y1, y2)

        return img[y1:y2, x1:x2, :]

    def pad_img_to_fit_bbox(self, img, x1, x2, y1, y2):
        img = cv2.copyMakeBorder(img, - min(0, y1),
                                 max(y2 - img.shape[0], 0),
                                 -min(0, x1), max(x2 - img.shape[1], 0),
                                 cv2.BORDER_REPLICATE
                                 )
        y2 += -min(0, y1)
        y1 += -min(0, y1)
        x2 += -min(0, x1)
        x1 += -min(0, x1)
        return img, x1, x2, y1, y2

    def crop_and_align(self, image_or_array=None, size=512, offset_x=0, offset_y=10, zoom_factor=2.0):
        img = None
        if isinstance(image_or_array, str):
            if not path.exists(image_or_array):
                message = "FaceTool: Image Path not found"
                return False, None,  message
            img = cv2.imread(image_or_array)
        elif type(image_or_array) is np.ndarray:
            img = image_or_array
        else:
            message = "FaceTool: Invalid Image"
            return 0, None, message

        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        face_results = self.face_detection.process(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Defining and drawing the rectangle around the face
        if not face_results.detections:
            message = "FaceTool: Face not found"
            return 0, None, message
        max_area = 0
        faces = []
        image_height, image_width = img.shape[:2]
        for face_index, face_result in enumerate(face_results.detections):
            face = Face()
            face.x = int(
                face_result.location_data.relative_bounding_box.xmin * image_width)
            face.y = int(
                face_result.location_data.relative_bounding_box.ymin * image_height)
            face.width = int(
                face_result.location_data.relative_bounding_box.width * image_width)
            face.height = int(
                face_result.location_data.relative_bounding_box.height * image_height)

            face.right_eye = (int(face_result.location_data.relative_keypoints[0].x * image_width), int(
                face_result.location_data.relative_keypoints[0].y * image_height))
            face.left_eye = (int(face_result.location_data.relative_keypoints[1].x * image_width), int(
                face_result.location_data.relative_keypoints[1].y * image_height))

            face.area = face.width * face.height
            if(face.area > max_area):
                max_area = face.area
                large_face_index = face_index

            # if self.debug:
            #     cv2.rectangle(img, (face.x, face.y), (face.x +
            #                   face.width, face.y + face.height), (0, 255, 0), 3)

            if(face.right_eye != None and face.left_eye != None):

                left_eye_x = face.left_eye[0]
                left_eye_y = face.left_eye[1]

                right_eye_x = face.right_eye[0]
                right_eye_y = face.right_eye[1]

                if self.debug:
                    cv2.circle(img, face.right_eye, 5, (255, 0, 0), -1)
                    cv2.circle(img, face.left_eye, 5, (255, 0, 0), -1)
                    cv2.line(img, face.right_eye,
                             face.left_eye, (0, 200, 200), 3)

                    if left_eye_y > right_eye_y:
                        A = (right_eye_x, left_eye_y)
                    else:
                        A = (left_eye_x, right_eye_y)

                    cv2.circle(img, A, 5, (255, 0, 0), -1)
                    cv2.line(img, face.right_eye,
                             face.left_eye, (0, 200, 200), 3)
                    cv2.line(img, face.left_eye, A, (0, 200, 100), 3)
                    cv2.line(img, face.right_eye, A, (0, 200, 100), 3)

                    # cv2.imwrite('roi_' + str(face_index) + '.png', img)
                    # cv2.imshow('image', img)
                    # cv2.waitKey(0)

                delta_x = right_eye_x - left_eye_x
                delta_y = right_eye_y - left_eye_y
                angle = np.arctan(delta_y/delta_x)
                angle = (angle * 180) / np.pi

                pointsToTransform = np.float32([[
                    [right_eye_x, right_eye_y],
                    [left_eye_x, left_eye_y],
                ]])

                # Calculating a center point of the image
                # Integer division "//"" ensures that we receive whole numbers
                center = (image_width // 2, image_height // 2)

                # Defining a matrix M and calling
                # cv2.getRotationMatrix2D method
                M = cv2.getRotationMatrix2D(center, (angle), 1.0)

                # Applying the rotation to our image using the
                # cv2.warpAffine method
                rotated = cv2.warpAffine(img, M, (image_width, image_height))

                # cv2.imwrite('rotated_' + str(face_index) + '.png', rotated)
                # cv2.imshow("rotated",rotated)
                # cv2.waitKey(0)

                transformedPoints = cv2.transform(pointsToTransform, M)[0]

                transformed_right_eye = transformedPoints[0]
                transformed_left_eye = transformedPoints[1]

                transformed_mid_point_x = int(transformed_left_eye[0] + (
                    transformed_right_eye[0] - transformed_left_eye[0])/2) + int(offset_x)
                transformed_mid_point_y = int(transformed_left_eye[1] + (
                    transformed_right_eye[1] - transformed_left_eye[1])/2) + int(offset_y)

                padding = int((transformed_right_eye[0] -
                               transformed_left_eye[0]) * zoom_factor)

                if self.debug:
                    cv2.circle(rotated, (transformed_mid_point_x,
                                         transformed_mid_point_y), 5, (96, 0, 90), -1)
                    cv2.circle(rotated, tuple(transformed_right_eye),
                               5, (255, 255, 0), -1)
                    cv2.circle(rotated, tuple(transformed_left_eye),
                               5, (255, 150, 0), -1)
                    cv2.rectangle(rotated, (transformed_mid_point_x - padding, transformed_mid_point_y - padding),
                                  (transformed_mid_point_x + padding, transformed_mid_point_y + padding), (0, 255, 0), 3)

                aligned = self.imcrop(rotated, [
                    transformed_mid_point_x - padding,
                    transformed_mid_point_y - padding,
                    transformed_mid_point_x + padding,
                    transformed_mid_point_y + padding,
                ])
                # cv2.imwrite('aligned_' + str(face_index) + '.png', rotated)
            else:
                padding = face.width//2.5 * zoom_factor
                midpoint_x = face.x + face.width//2
                midpoint_y = face.y + face.height//2
                aligned = self.imcrop(img, [
                    midpoint_x - padding,
                    midpoint_y - padding,
                    midpoint_x + padding,
                    midpoint_y + padding,
                ])
            if(aligned.shape[0] <= 0 or aligned.shape[1] <= 0):
                print("Failed to generate image")
            else:
                face.image = cv2.resize(aligned, (size, size))
                faces.append(face)
        # sort the faces by area.
        faces.sort(key=lambda x: x.area, reverse=True)
        return len(face_results.detections), faces, "Success"


if __name__ == "__main__":
    facetool = FaceTool(debug=False)
    videoMode = True
    path = "samples"
    if videoMode:
        vid = cv2.VideoCapture(0)
        while True:
            ret, image = vid.read()
            face_count, faces, message = facetool.crop_and_align(image)
            if face_count > 0:
                cv2.imwrite(path + "/output/video.png", faces[0].image)
                cv2.imshow("aligned_image", faces[0].image)
            else:
                print(message)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        vid.release()
    else:
        # path = os.path.dirname(os.path.abspath(__file__)) + "/datasets/facetool"
        onlyfiles = [(f, join(path, f))
                     for f in listdir(path) if not f.startswith('.') and isfile(join(path, f))]
        total = len(onlyfiles)
        counter = 0
        for filePath in onlyfiles:
            print(filePath[1])
            image = cv2.imread(filePath[1])
            face_count, faces, message = facetool.crop_and_align(
                image, size=256, zoom_factor=2, offset_y=-10)
            if face_count > 0:
                # face are sorted by area
                # takes the face with largest area.
                cv2.imwrite(path + "/output/" + filePath[0], faces[0].image)
                cv2.imshow("aligned_image", faces[0].image)
            else:
                print(message)
            key = cv2.waitKey(1000) & 0xFFa
            if key == ord("q"):
                break
    cv2.destroyAllWindows()
