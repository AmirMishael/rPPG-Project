import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(r'C:/Users/amir9/OneDrive/Desktop/Technion_Bsc/7th_semester/Project_A_44167/Python_Files/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(r'C:/Users/amir9/OneDrive/Desktop/Technion_Bsc/7th_semester/Project_A_44167/Python_Files/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)  # "C:/Users/amir9/OneDrive/Desktop/Technion_Bsc/7th_semester/Project_A_44167/vid.avi"

ret_init, Initialization_Frame = cap.read()

Faces_Backup = [0, 0, Initialization_Frame[0, :].size, Initialization_Frame[:, 1].size]  # Backup array of face location parameters (x,y,w,h) in case the face is not detected in a frame
Eyes_Backup = np.zeros((2, 4))  # Backup array of eyes location parameters (x,y,w,h) in case eyes are not detected in a frame

while True:
    ret, img = cap.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, 1.25, 4)
    eyes = eye_cascade.detectMultiScale(gray_img)
    if not type(faces) == tuple:
        Faces_Backup[0] = faces[0, 0]
        Faces_Backup[1] = faces[0, 1]
        Faces_Backup[2] = faces[0, 2]
        Faces_Backup[3] = faces[0, 3]
        if not type(eyes) == tuple:
            if eyes[:, 0].size > 1:
                Eyes_Backup[0, 0] = eyes[0, 0]
                Eyes_Backup[0, 1] = eyes[0, 1]
                Eyes_Backup[0, 2] = eyes[0, 2]
                Eyes_Backup[0, 3] = eyes[0, 3]
                Eyes_Backup[1, 0] = eyes[1, 0]
                Eyes_Backup[1, 1] = eyes[1, 1]
                Eyes_Backup[1, 2] = eyes[1, 2]
                Eyes_Backup[1, 3] = eyes[1, 3]
            else:
                Eyes_Backup[0, 0] = eyes[0, 0]
                Eyes_Backup[0, 1] = eyes[0, 1]
                Eyes_Backup[0, 2] = eyes[0, 2]
                Eyes_Backup[0, 3] = eyes[0, 3]

    try:
        cv2.rectangle(img, (faces[0, 0], faces[0, 1]), (faces[0, 0] + faces[0, 2], faces[0, 1]+faces[0, 3]), (255, 255, 0), 2)
        cv2.rectangle(img, (int(Eyes_Backup[0, 0]), int(Eyes_Backup[0, 1])), (int(Eyes_Backup[0, 0] + Eyes_Backup[0, 2]), int(Eyes_Backup[0, 1]+Eyes_Backup[0, 3])), (0, 127, 255), 2)
        cv2.rectangle(img, (int(Eyes_Backup[1, 0]), int(Eyes_Backup[1, 1])), (int(Eyes_Backup[1, 0] + Eyes_Backup[1, 2]), int(Eyes_Backup[1, 1]+Eyes_Backup[1, 3])), (0, 127, 255), 2)
        cv2.rectangle(img, (int(np.minimum(Eyes_Backup[0, 0], Eyes_Backup[1, 0])), int(np.minimum(Eyes_Backup[0, 1]-Eyes_Backup[0, 3], Eyes_Backup[1, 1]-Eyes_Backup[1, 3]))), (int(np.maximum(Eyes_Backup[0, 0]+Eyes_Backup[0, 2], Eyes_Backup[1, 0]+Eyes_Backup[1, 2])), int(np.minimum(faces[0, 1] + faces[0, 3], Eyes_Backup[0, 1]))), (255, 0, 255), 2)
        cv2.rectangle(img, (Faces_Backup[0], Faces_Backup[1]), (Faces_Backup[0] + Faces_Backup[2], Faces_Backup[1]+Faces_Backup[3]), (210, 210, 210), 2)
    except NameError:
        cv2.rectangle(img, (faces[0, 0], faces[0, 1]), (faces[0, 0] + faces[0, 2], faces[0, 1]+faces[0, 3]), (255, 255, 0), 2)
        cv2.rectangle(img, (int(Eyes_Backup[0, 0]), int(Eyes_Backup[0, 1])), (int(Eyes_Backup[0, 0] + Eyes_Backup[0, 2]), int(Eyes_Backup[0, 1]+Eyes_Backup[0, 3])), (0, 127, 255), 2)
        cv2.rectangle(img, (int(Eyes_Backup[0, 0]), int(Eyes_Backup[0, 1]-Eyes_Backup[0, 3])), (int(Eyes_Backup[0, 0]+3*Eyes_Backup[0, 2]), int(np.minimum(faces[0, 1] + faces[0, 3], Eyes_Backup[0, 1]))), (255, 0, 255), 2)
        cv2.rectangle(img, (Faces_Backup[0], Faces_Backup[1]), (Faces_Backup[0] + Faces_Backup[2], Faces_Backup[1]+Faces_Backup[3]), (210, 210, 210), 2)
    except:
        cv2.rectangle(img, (Faces_Backup[0], Faces_Backup[1]), (Faces_Backup[0] + Faces_Backup[2], Faces_Backup[1]+Faces_Backup[3]), (210, 210, 210), 2)
        # print("missing face parameters")

    cv2.imshow('Face Recognition', img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
