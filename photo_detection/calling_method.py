import base64

from photo_detection.faceDetect_photo import is_smiling

# writing jpg as string
with open("./data/images/smiling_person3.jpg", "rb") as img_file:
    photo_string = base64.b64encode(img_file.read())


# print(faceDetect_photo.is_smiling('smiling_girl.jpg'))
# print(faceDetect_photo.is_smiling('smiling_dude.jpg'))
# print(faceDetect_photo.is_smiling('smiling_person.jpg'))
# print(faceDetect_photo.is_smiling('smiling_person2.jpg'))
# print(faceDetect_photo.is_smiling('smiling_person3.jpg'))
print(is_smiling(photo_string))
