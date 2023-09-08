# from mediapipe import solutions
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision

# from landmark_draw import draw_landmarks_on_image

# base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
# options = vision.HandLandmarkerOptions(base_options=base_options,
#                                        num_hands = 1)
# detector = vision.HandLandmarker.create_from_options(options)


# img = mp.Image.create_from_file("Image_1694073828.8708694.png")

# detection_result = detector.detect(img)

# annotated_img = draw_landmarks_on_image(img.numpy_view(), detection_result)
# cv2.imshow('', cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)