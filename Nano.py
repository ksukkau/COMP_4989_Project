import cv2

cap = cv2.VideoCapture(0)

if (cap.isOpened() == False):
    print("Unable to open baby camera for baby vid")


frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

while(True):
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break;

    else:
        break

cap.release()
cv2.destroyAllWindows()
