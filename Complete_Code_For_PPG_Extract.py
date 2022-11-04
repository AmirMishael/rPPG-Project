import cv2
import sys
import numpy as np

# import pandas as pd
# import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier(
    r'C:/Users/amir9/OneDrive/Desktop/Technion_Bsc/7th_semester/Project_A_44167/Python_Files/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    r'C:/Users/amir9/OneDrive/Desktop/Technion_Bsc/7th_semester/Project_A_44167/Python_Files/haarcascade_eye.xml')
np.set_printoptions(threshold=sys.maxsize)
cap = cv2.VideoCapture("C:/Users/amir9/OneDrive/Desktop/Technion_Bsc/7th_semester/Project_A_44167/vid.avi")
Frame_Num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
FPS = int(cap.get(cv2.CAP_PROP_FPS))

ret_init, Initialization_Frame = cap.read()

Faces_Backup = [0, 0, Initialization_Frame[0, :].size, Initialization_Frame[:, 1].size]  # Backup array of face location parameters (x,y,w,h) in case the face is not detected in a frame
Eyes_Backup = np.zeros((2, 4))  # Backup array of eyes location parameters (x,y,w,h) in case eyes are not detected in a frame
left = right = top = bottom = 0

# forehead_signal = np.zeros((Frame_Num, 9))
# Creating a complete 2D array for forehead signal. The order of the data is HSV, YCbCr, global and subdivision RGB for all three. All in all Frame_Num rows over 9 columns.
# Same may be done for the total face signal.
tot_Blue_Signal_HSV = np.zeros(Frame_Num)
tot_Green_Signal_HSV = np.zeros(Frame_Num)
tot_Red_Signal_HSV = np.zeros(Frame_Num)
tot_Blue_Signal_YCbCr = np.zeros(Frame_Num)
tot_Green_Signal_YCbCr = np.zeros(Frame_Num)
tot_Red_Signal_YCbCr = np.zeros(Frame_Num)
tot_Blue_Signal_global = np.zeros(Frame_Num)
tot_Green_Signal_global = np.zeros(Frame_Num)
tot_Red_Signal_global = np.zeros(Frame_Num)

forehead_Blue_Signal_HSV = np.zeros(Frame_Num)
forehead_Green_Signal_HSV = np.zeros(Frame_Num)
forehead_Red_Signal_HSV = np.zeros(Frame_Num)
forehead_Blue_Signal_YCbCr = np.zeros(Frame_Num)
forehead_Green_Signal_YCbCr = np.zeros(Frame_Num)
forehead_Red_Signal_YCbCr = np.zeros(Frame_Num)
forehead_Blue_Signal_global = np.zeros(Frame_Num)
forehead_Green_Signal_global = np.zeros(Frame_Num)
forehead_Red_Signal_global = np.zeros(Frame_Num)

for f_ind in range(0, Frame_Num - 1):
    s_HSV = p_HSV = s_YCbCr = p_YCbCr = s_global = p_global = 0  # s indices represent forehead summation and p indices represent total face summation
    forehead_skin_pixel_sum_B_HSV = forehead_skin_pixel_sum_G_HSV = forehead_skin_pixel_sum_R_HSV = tot_skin_pixel_sum_B_HSV = tot_skin_pixel_sum_G_HSV = tot_skin_pixel_sum_R_HSV = 0.000001
    forehead_skin_pixel_sum_B_YCbCr = forehead_skin_pixel_sum_G_YCbCr = forehead_skin_pixel_sum_R_YCbCr = tot_skin_pixel_sum_B_YCbCr = tot_skin_pixel_sum_G_YCbCr = tot_skin_pixel_sum_R_YCbCr = 0.000001
    forehead_skin_pixel_sum_B_global = forehead_skin_pixel_sum_G_global = forehead_skin_pixel_sum_R_global = tot_skin_pixel_sum_B_global = tot_skin_pixel_sum_G_global = tot_skin_pixel_sum_R_global = 0.000001

    ret, img = cap.read()
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

    eyes = eye_cascade.detectMultiScale(gray_img)

    try:  # forehead data extract when 2 eyes are clearly detected
        Eyes_Backup[0, 0] = eyes[0, 0]
        Eyes_Backup[0, 1] = eyes[0, 1]
        Eyes_Backup[0, 2] = eyes[0, 2]
        Eyes_Backup[0, 3] = eyes[0, 3]
        Eyes_Backup[1, 0] = eyes[1, 0]
        Eyes_Backup[1, 1] = eyes[1, 1]
        Eyes_Backup[1, 2] = eyes[1, 2]
        Eyes_Backup[1, 3] = eyes[1, 3]
        for y in range(np.minimum(eyes[0, 0], eyes[1, 0]),
                       np.maximum(eyes[0, 0] + eyes[0, 2], eyes[1, 0] + eyes[1, 2])):
            for x in range(np.minimum(eyes[0, 1] - eyes[0, 3], eyes[1, 1] - eyes[1, 3]),
                           np.minimum(bottom, eyes[0, 1])):
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
            forehead_Blue_Signal_YCbCr[f_ind] = forehead_skin_pixel_sum_B_YCbCr / s_YCbCr
            forehead_Green_Signal_YCbCr[f_ind] = forehead_skin_pixel_sum_G_YCbCr / s_YCbCr
            forehead_Red_Signal_YCbCr[f_ind] = forehead_skin_pixel_sum_R_YCbCr / s_YCbCr

        except:
            pass

        try:
            forehead_Blue_Signal_HSV[f_ind] = forehead_skin_pixel_sum_B_HSV / s_HSV
            forehead_Green_Signal_HSV[f_ind] = forehead_skin_pixel_sum_G_HSV / s_HSV
            forehead_Red_Signal_HSV[f_ind] = forehead_skin_pixel_sum_R_HSV / s_HSV

        except:
            pass

        try:
            forehead_Blue_Signal_global[f_ind] = forehead_skin_pixel_sum_B_global / s_global
            forehead_Green_Signal_global[f_ind] = forehead_skin_pixel_sum_G_global / s_global
            forehead_Red_Signal_global[f_ind] = forehead_skin_pixel_sum_R_global / s_global

        except:
            pass

    except NameError:  # forehead data extract when only one eye is clearly detected
        Eyes_Backup[0, 0] = eyes[0, 0]
        Eyes_Backup[0, 1] = eyes[0, 1]
        Eyes_Backup[0, 2] = eyes[0, 2]
        Eyes_Backup[0, 3] = eyes[0, 3]
        for y in range(eyes[0, 0], eyes[0, 0] + 3 * eyes[0, 2]):
            for x in range(eyes[0, 1] - eyes[0, 3], np.minimum(bottom, eyes[0, 1])):
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
            forehead_Blue_Signal_YCbCr[f_ind] = forehead_skin_pixel_sum_B_YCbCr / s_YCbCr
            forehead_Green_Signal_YCbCr[f_ind] = forehead_skin_pixel_sum_G_YCbCr / s_YCbCr
            forehead_Red_Signal_YCbCr[f_ind] = forehead_skin_pixel_sum_R_YCbCr / s_YCbCr

        except:
            pass

        try:
            forehead_Blue_Signal_HSV[f_ind] = forehead_skin_pixel_sum_B_HSV / s_HSV
            forehead_Green_Signal_HSV[f_ind] = forehead_skin_pixel_sum_G_HSV / s_HSV
            forehead_Red_Signal_HSV[f_ind] = forehead_skin_pixel_sum_R_HSV / s_HSV

        except:
            pass

        try:
            forehead_Blue_Signal_global[f_ind] = forehead_skin_pixel_sum_B_global / s_global
            forehead_Green_Signal_global[f_ind] = forehead_skin_pixel_sum_G_global / s_global
            forehead_Red_Signal_global[f_ind] = forehead_skin_pixel_sum_R_global / s_global

        except:
            pass

    except ConnectionError:
        for y in range(np.minimum(Eyes_Backup[0, 0], Eyes_Backup[1, 0]),
                         np.maximum(Eyes_Backup[0, 0] + Eyes_Backup[0, 2], Eyes_Backup[1, 0] + Eyes_Backup[1, 2])):
            for x in range(np.minimum(Eyes_Backup[0, 1] - Eyes_Backup[0, 3], Eyes_Backup[1, 1] - Eyes_Backup[1, 3]),
                              np.minimum(bottom, Eyes_Backup[0, 1])):
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
            forehead_Blue_Signal_YCbCr[f_ind] = forehead_skin_pixel_sum_B_YCbCr / s_YCbCr
            forehead_Green_Signal_YCbCr[f_ind] = forehead_skin_pixel_sum_G_YCbCr / s_YCbCr
            forehead_Red_Signal_YCbCr[f_ind] = forehead_skin_pixel_sum_R_YCbCr / s_YCbCr

        except:
            pass

        try:
            forehead_Blue_Signal_HSV[f_ind] = forehead_skin_pixel_sum_B_HSV / s_HSV
            forehead_Green_Signal_HSV[f_ind] = forehead_skin_pixel_sum_G_HSV / s_HSV
            forehead_Red_Signal_HSV[f_ind] = forehead_skin_pixel_sum_R_HSV / s_HSV

        except:
            pass

        try:
            forehead_Blue_Signal_global[f_ind] = forehead_skin_pixel_sum_B_global / s_global
            forehead_Green_Signal_global[f_ind] = forehead_skin_pixel_sum_G_global / s_global
            forehead_Red_Signal_global[f_ind] = forehead_skin_pixel_sum_R_global / s_global

        except:
            pass

        print("Something unknown went wrong")
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
        tot_Blue_Signal_YCbCr[f_ind] = tot_skin_pixel_sum_B_YCbCr / p_YCbCr
        tot_Green_Signal_YCbCr[f_ind] = tot_skin_pixel_sum_G_YCbCr / p_YCbCr
        tot_Red_Signal_YCbCr[f_ind] = tot_skin_pixel_sum_R_YCbCr / p_YCbCr

    except:
        pass

    try:
        tot_Blue_Signal_HSV[f_ind] = tot_skin_pixel_sum_B_HSV / p_HSV
        tot_Green_Signal_HSV[f_ind] = tot_skin_pixel_sum_G_HSV / p_HSV
        tot_Red_Signal_HSV[f_ind] = tot_skin_pixel_sum_R_HSV / p_HSV

    except:
        pass

    try:
        tot_Blue_Signal_global[f_ind] = tot_skin_pixel_sum_B_global / p_global
        tot_Green_Signal_global[f_ind] = tot_skin_pixel_sum_G_global / p_global
        tot_Red_Signal_global[f_ind] = tot_skin_pixel_sum_R_global / p_global

    except:
        pass
    formatted_progress = "{:.2f}".format(100 * (f_ind / Frame_Num))
    print("we are at frame", f_ind, "of", Frame_Num, "which means progress of", formatted_progress, "%\n")
    # Eyes_Backup[0, 0] = Eyes_Backup[1, 0] = 0
    # Eyes_Backup[0, 2] = Eyes_Backup[1, 2] = gray_img[0, :].size
    # Eyes_Backup[0, 3] = Eyes_Backup[1, 3] = gray_img[:, 1].size

np.savetxt("forehead_Red_HSV_Backups.csv", forehead_Red_Signal_YCbCr, delimiter=",")
