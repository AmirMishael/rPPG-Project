import cv2
import sys
import numpy as np
import keyboard

# import pandas as pd
# import matplotlib.pyplot as plt


face_cascade = cv2.CascadeClassifier(
    r'C:/Users/amir9/OneDrive/Desktop/Technion_Bsc/7th_semester/Project_A_44167/Python_Files/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    r'C:/Users/amir9/OneDrive/Desktop/Technion_Bsc/7th_semester/Project_A_44167/Python_Files/haarcascade_eye.xml')
np.set_printoptions(threshold=sys.maxsize)
cap = cv2.VideoCapture(0)
# Frame_Num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# FPS = int(cap.get(cv2.CAP_PROP_FPS))
# print(FPS)

ret_init, Initialization_Frame = cap.read()

Faces_Backup = [0, 0, Initialization_Frame[0, :].size, Initialization_Frame[:,
                                                       1].size]  # Backup array of face location parameters (x,y,w,h) in case the face is not detected in a frame
Eyes_Backup = np.zeros(
    (2, 4))  # Backup array of eyes location parameters (x,y,w,h) in case eyes are not detected in a frame
left = right = top = bottom = forehead_left = forehead_right = forehead_top = forehead_bottom = 0
f_ind = 0
n = 400  # wanted number of frames to have in the signal that will represent about 10 seconds
Signal = np.zeros(n)

# forehead_signal = np.zeros((Frame_Num, 9))
# Creating a complete 2D array for forehead signal. The order of the data is HSV, YCbCr, global and subdivision RGB for all three. All in all Frame_Num rows over 9 columns.
# Same may be done for the total face signal.
live_tot_Blue_Signal_HSV = np.array([])
live_tot_Green_Signal_HSV = np.array([])
live_tot_Red_Signal_HSV = np.array([])
live_tot_Blue_Signal_YCbCr = np.array([])
live_tot_Green_Signal_YCbCr = np.array([])
live_tot_Red_Signal_YCbCr = np.array([])
live_tot_Blue_Signal_global = np.array([])
live_tot_Green_Signal_global = np.array([])
live_tot_Red_Signal_global = np.array([])

live_forehead_Blue_Signal_HSV = np.array([])
live_forehead_Green_Signal_HSV = np.array([])
live_forehead_Red_Signal_HSV = np.array([])
live_forehead_Blue_Signal_YCbCr = np.array([])
live_forehead_Green_Signal_YCbCr = np.array([])
live_forehead_Red_Signal_YCbCr = np.array([])
live_forehead_Blue_Signal_global = np.array([])
live_forehead_Green_Signal_global = np.array([])
live_forehead_Red_Signal_global = np.array([])

# while True:
while not keyboard.is_pressed(True):
    s_HSV = p_HSV = s_YCbCr = p_YCbCr = s_global = p_global = 0  # s indices represent forehead summation and p indices represent total face summation
    forehead_skin_pixel_sum_B_HSV = forehead_skin_pixel_sum_G_HSV = forehead_skin_pixel_sum_R_HSV = tot_skin_pixel_sum_B_HSV = tot_skin_pixel_sum_G_HSV = tot_skin_pixel_sum_R_HSV = 0.000001
    forehead_skin_pixel_sum_B_YCbCr = forehead_skin_pixel_sum_G_YCbCr = forehead_skin_pixel_sum_R_YCbCr = tot_skin_pixel_sum_B_YCbCr = tot_skin_pixel_sum_G_YCbCr = tot_skin_pixel_sum_R_YCbCr = 0.000001
    forehead_skin_pixel_sum_B_global = forehead_skin_pixel_sum_G_global = forehead_skin_pixel_sum_R_global = tot_skin_pixel_sum_B_global = tot_skin_pixel_sum_G_global = tot_skin_pixel_sum_R_global = 0.000001

    ret, img = cap.read()
    f_ind += 1  # frames index, counting frames

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    frame_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    HSV_mask = cv2.inRange(frame_HSV, (0, 15, 0), (17, 170, 255))
    HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    HSV_result = cv2.bitwise_not(HSV_mask)

    frame_YCbCr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    YCbCr_mask = cv2.inRange(frame_YCbCr, (0, 135, 85), (255, 180, 135))
    YCbCr_mask = cv2.morphologyEx(YCbCr_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    YCbCr_result = cv2.bitwise_not(YCbCr_mask)

    global_mask = cv2.bitwise_and(YCbCr_mask, HSV_mask)
    global_mask = cv2.medianBlur(global_mask, 3)
    global_mask = cv2.morphologyEx(global_mask, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))
    global_result = cv2.bitwise_not(global_mask)

    faces = face_cascade.detectMultiScale(gray_img, 1.25, 4)
    if not type(faces) == tuple:
        right = faces[0, 0] + faces[0, 2]
        left = faces[0, 0]
        top = faces[0, 1]
        bottom = faces[0, 1] + faces[0, 3]
        Faces_Backup[0] = faces[0, 0]
        Faces_Backup[1] = faces[0, 1]
        Faces_Backup[2] = faces[0, 2]
        Faces_Backup[3] = faces[0, 3]
    else:
        right = Faces_Backup[0] + Faces_Backup[2]
        left = Faces_Backup[0]
        top = Faces_Backup[1]
        bottom = Faces_Backup[1] + Faces_Backup[3]
    # print("right = ", right, "left = ", left, "top = ", top, "bottom = ", bottom)
    eyes = eye_cascade.detectMultiScale(gray_img)
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
            forehead_left = np.minimum(eyes[0, 0], eyes[1, 0])
            forehead_right = np.maximum(eyes[0, 0] + eyes[0, 2], eyes[1, 0] + eyes[1, 2])
            forehead_top = np.minimum(eyes[0, 1] - eyes[0, 3], eyes[1, 1] - eyes[1, 3])
            forehead_bottom = np.minimum(bottom, eyes[0, 1])
        else:
            Eyes_Backup[0, 0] = eyes[0, 0]
            Eyes_Backup[0, 1] = eyes[0, 1]
            Eyes_Backup[0, 2] = eyes[0, 2]
            Eyes_Backup[0, 3] = eyes[0, 3]
            forehead_left = eyes[0, 0]
            forehead_right = eyes[0, 0] + 3 * eyes[0, 2]
            forehead_top = eyes[0, 1] - eyes[0, 3]
            forehead_bottom = np.minimum(bottom, eyes[0, 1])
    else:
        forehead_left = int(np.minimum(Eyes_Backup[0, 0], Eyes_Backup[1, 0]))
        forehead_right = int(np.maximum(Eyes_Backup[0, 0] + Eyes_Backup[0, 2], Eyes_Backup[1, 0] + Eyes_Backup[1, 2]))
        forehead_top = int(np.minimum(Eyes_Backup[0, 1] - Eyes_Backup[0, 3], Eyes_Backup[1, 1] - Eyes_Backup[1, 3]))
        forehead_bottom = int(np.minimum(bottom, Eyes_Backup[0, 1]))

    # try:  # forehead data extract when 2 eyes are clearly detected
    for y in range(int(forehead_left),
                   int(forehead_right)):
        for x in range(int(forehead_top),
                       int(forehead_bottom)):
            if not YCbCr_result[x][y]:  # skin pixel
                forehead_skin_pixel_sum_B_YCbCr += img[x][y][0]
                forehead_skin_pixel_sum_G_YCbCr += img[x][y][1]
                forehead_skin_pixel_sum_R_YCbCr += img[x][y][2]
                s_YCbCr += 1

            if not HSV_result[x][y]:  # skin pixel
                forehead_skin_pixel_sum_B_HSV += img[x][y][0]
                forehead_skin_pixel_sum_G_HSV += img[x][y][1]
                forehead_skin_pixel_sum_R_HSV += img[x][y][2]
                s_HSV += 1

            if not global_result[x][y]:  # skin pixel
                forehead_skin_pixel_sum_B_global += img[x][y][0]
                forehead_skin_pixel_sum_G_global += img[x][y][1]
                forehead_skin_pixel_sum_R_global += img[x][y][2]
                s_global += 1

    try:
        live_forehead_Blue_Signal_YCbCr = np.append(live_forehead_Blue_Signal_YCbCr,
                                                    forehead_skin_pixel_sum_B_YCbCr / s_YCbCr)
        live_forehead_Green_Signal_YCbCr = np.append(live_forehead_Green_Signal_YCbCr,
                                                     forehead_skin_pixel_sum_G_YCbCr / s_YCbCr)
        live_forehead_Red_Signal_YCbCr = np.append(live_forehead_Red_Signal_YCbCr,
                                                   forehead_skin_pixel_sum_R_YCbCr / s_YCbCr)

    except:  # data wasn't collectable, therefore we append 0
        live_forehead_Blue_Signal_YCbCr = np.append(live_forehead_Blue_Signal_YCbCr, 0)
        live_forehead_Green_Signal_YCbCr = np.append(live_forehead_Green_Signal_YCbCr, 0)
        live_forehead_Red_Signal_YCbCr = np.append(live_forehead_Red_Signal_YCbCr, 0)

    try:
        live_forehead_Blue_Signal_HSV = np.append(live_forehead_Blue_Signal_HSV,
                                                  forehead_skin_pixel_sum_B_HSV / s_HSV)
        live_forehead_Green_Signal_HSV = np.append(live_forehead_Green_Signal_HSV,
                                                   forehead_skin_pixel_sum_G_HSV / s_HSV)
        live_forehead_Red_Signal_HSV = np.append(live_forehead_Red_Signal_HSV,
                                                 forehead_skin_pixel_sum_R_HSV / s_HSV)

    except:
        live_forehead_Blue_Signal_HSV = np.append(live_forehead_Blue_Signal_HSV, 0)
        live_forehead_Green_Signal_HSV = np.append(live_forehead_Green_Signal_HSV, 0)
        live_forehead_Red_Signal_HSV = np.append(live_forehead_Red_Signal_HSV, 0)

    try:
        live_forehead_Blue_Signal_global = np.append(live_forehead_Blue_Signal_global,
                                                     forehead_skin_pixel_sum_B_global / s_global)
        live_forehead_Green_Signal_global = np.append(live_forehead_Green_Signal_global,
                                                      forehead_skin_pixel_sum_G_global / s_global)
        live_forehead_Red_Signal_global = np.append(live_forehead_Red_Signal_global,
                                                    forehead_skin_pixel_sum_R_global / s_global)

    except:
        live_forehead_Blue_Signal_global = np.append(live_forehead_Blue_Signal_global, 0)
        live_forehead_Green_Signal_global = np.append(live_forehead_Green_Signal_global, 0)
        live_forehead_Red_Signal_global = np.append(live_forehead_Red_Signal_global, 0)

    # going generally over the face
    for x in range(top, bottom):
        for y in range(left, right):
            if not YCbCr_result[x][y]:  # skin pixel
                tot_skin_pixel_sum_B_YCbCr += img[x][y][0]
                tot_skin_pixel_sum_G_YCbCr += img[x][y][1]
                tot_skin_pixel_sum_R_YCbCr += img[x][y][2]
                p_YCbCr += 1

            if not HSV_result[x][y]:  # skin pixel
                tot_skin_pixel_sum_B_HSV += img[x][y][0]
                tot_skin_pixel_sum_G_HSV += img[x][y][1]
                tot_skin_pixel_sum_R_HSV += img[x][y][2]
                p_HSV += 1

            if not global_result[x][y]:  # skin pixel
                tot_skin_pixel_sum_B_global += img[x][y][0]
                tot_skin_pixel_sum_G_global += img[x][y][1]
                tot_skin_pixel_sum_R_global += img[x][y][2]
                p_global += 1

    try:
        live_tot_Blue_Signal_YCbCr = np.append(live_tot_Blue_Signal_YCbCr, tot_skin_pixel_sum_B_YCbCr / p_YCbCr)
        live_tot_Green_Signal_YCbCr = np.append(live_tot_Green_Signal_YCbCr, tot_skin_pixel_sum_G_YCbCr / p_YCbCr)
        live_tot_Red_Signal_YCbCr = np.append(live_tot_Red_Signal_YCbCr, tot_skin_pixel_sum_R_YCbCr / p_YCbCr)

    except:
        tot_Blue_Signal_YCbCr = np.append(live_tot_Blue_Signal_YCbCr, 0)
        live_tot_Green_Signal_YCbCr = np.append(live_tot_Green_Signal_YCbCr, 0)
        live_tot_Red_Signal_YCbCr = np.append(live_tot_Red_Signal_YCbCr, 0)

    try:
        live_tot_Blue_Signal_HSV = np.append(live_tot_Blue_Signal_HSV, tot_skin_pixel_sum_B_HSV / p_HSV)
        live_tot_Green_Signal_HSV = np.append(live_tot_Green_Signal_HSV, tot_skin_pixel_sum_G_HSV / p_HSV)
        live_tot_Red_Signal_HSV = np.append(live_tot_Red_Signal_HSV, tot_skin_pixel_sum_R_HSV / p_HSV)

    except:
        live_tot_Blue_Signal_HSV = np.append(live_tot_Blue_Signal_HSV, 0)
        live_tot_Green_Signal_HSV = np.append(live_tot_Green_Signal_HSV, 0)
        live_tot_Red_Signal_HSV = np.append(live_tot_Red_Signal_HSV, 0)

    try:
        live_tot_Blue_Signal_global = np.append(live_tot_Blue_Signal_global, tot_skin_pixel_sum_B_global / p_global)
        live_tot_Green_Signal_global = np.append(live_tot_Green_Signal_global, tot_skin_pixel_sum_G_global / p_global)
        live_tot_Red_Signal_global = np.append(live_tot_Red_Signal_global, tot_skin_pixel_sum_R_global / p_global)

    except:
        live_tot_Blue_Signal_global = np.append(live_tot_Blue_Signal_global, 0)
        live_tot_Green_Signal_global = np.append(live_tot_Green_Signal_global, 0)
        live_tot_Red_Signal_global = np.append(live_tot_Red_Signal_global, 0)
    # try:
    #     cv2.rectangle(img, (faces[0, 0], faces[0, 1]), (faces[0, 0] + faces[0, 2], faces[0, 1]+faces[0, 3]), (255, 255, 0), 2)
    #     cv2.rectangle(img, (eyes[0, 0], eyes[0, 1]), (eyes[0, 0] + eyes[0, 2], eyes[0, 1]+eyes[0, 3]), (0, 127, 255), 2)
    #     cv2.rectangle(img, (eyes[1, 0], eyes[1, 1]), (eyes[1, 0] + eyes[1, 2], eyes[1, 1]+eyes[1, 3]), (0, 127, 255), 2)
    #     cv2.rectangle(img, (np.minimum(eyes[0, 0], eyes[1, 0]), np.minimum(eyes[0, 1]-eyes[0, 3], eyes[1, 1]-eyes[1, 3])), (np.maximum(eyes[0, 0]+eyes[0, 2], eyes[1, 0]+eyes[1, 2]), np.minimum(faces[0, 1] + faces[0, 3], eyes[0, 1])), (255, 0, 255), 2)
    #     cv2.rectangle(img, (Faces_Backup[0], Faces_Backup[1]), (Faces_Backup[0] + Faces_Backup[2], Faces_Backup[1]+Faces_Backup[3]), (210, 210, 210), 2)
    # except NotADirectoryError:
    #     cv2.rectangle(img, (faces[0, 0], faces[0, 1]), (faces[0, 0] + faces[0, 2], faces[0, 1]+faces[0, 3]), (255, 255, 0), 2)
    #     cv2.rectangle(img, (eyes[0, 0], eyes[0, 1]), (eyes[0, 0] + eyes[0, 2], eyes[0, 1]+eyes[0, 3]), (0, 127, 255), 2)
    #     cv2.rectangle(img, (eyes[0, 0], eyes[0, 1]-eyes[0, 3]), (eyes[0, 0]+3*eyes[0, 2], np.minimum(faces[0, 1] + faces[0, 3], eyes[0, 1])), (255, 0, 255), 2)
    #     cv2.rectangle(img, (Faces_Backup[0], Faces_Backup[1]), (Faces_Backup[0] + Faces_Backup[2], Faces_Backup[1]+Faces_Backup[3]), (210, 210, 210), 2)
    # except:
    #     cv2.rectangle(img, (Faces_Backup[0], Faces_Backup[1]), (Faces_Backup[0] + Faces_Backup[2], Faces_Backup[1]+Faces_Backup[3]), (210, 210, 210), 2)

    # c = cv2.waitKey(7) % 0x100
    # if c == 27 or c == 10:
    #     breakqq
    if keyboard.is_pressed("q"):
        # Key was pressed
        break
    print("we are at frame", f_ind)
    # k = keyboard.read_key()
    # if k == "esc":
    #     break
    # cv2.imshow("Generic name", img)
    if f_ind > n:
        Signal = live_forehead_Red_Signal_YCbCr[f_ind - n:f_ind]
    # k = cv2.waitKey(30) & 0xff
    # if k == 27:
    #     break

np.savetxt("live_forehead_Red_HSV_Backups.csv", Signal, delimiter=",")
