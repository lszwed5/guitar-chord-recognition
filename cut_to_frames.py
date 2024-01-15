import cv2
import time
cpt = 0
maxFrames = 300 # if you want 5 frames only.
count = 0


cap=cv2.VideoCapture('videos/None_close_up.mp4')
while cpt < maxFrames:
    ret, frame = cap.read()
    # count += 1
    # if count % 2 != 0:
    #     continue
    frame=cv2.resize(frame,(1080,500))
    time.sleep(0.01)
    frame=cv2.flip(frame,1)
    cv2.imshow("test window", frame) # show image in window
    cv2.imwrite(r"Mediapipe_data_close/None/%d.jpg" %cpt, frame)
    cpt += 1
    if cv2.waitKey(5)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
