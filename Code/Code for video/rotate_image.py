import imutils

def rotate(image, angle):
    rotated = imutils.rotate_bound(image, angle)

    return rotated

