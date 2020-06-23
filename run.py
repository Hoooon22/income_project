import cv2
from src.hand_tracker import HandTracker

WINDOW = "Hand Tracking"
PALM_MODEL_PATH = "models/palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = "models/hand_landmark.tflite"
ANCHORS_PATH = "models/anchors.csv"

POINT_COLOR = (0, 255, 0)
CONNECTION_COLOR = (255, 0, 0)
THICKNESS = 2

cv2.namedWindow(WINDOW)
capture = cv2.VideoCapture(0)

if capture.isOpened():
    hasFrame, frame = capture.read()
else:
    hasFrame = False

#        8   12  16  20
#        |   |   |   |
#        7   11  15  19
#    4   |   |   |   |
#    |   6   10  14  18
#    3   |   |   |   |
#    |   5---9---13--17
#    2    \         /
#     \    \       /
#      1    \     /
#       \    \   /
#        ------0-
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
]

detector = HandTracker(
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    box_shift=0.2,
    box_enlarge=1.3
)
# need adding
live_count = 0
pre_num = [0, 0, 0, 0, 0]
breaker = 0
sum = 0

while hasFrame:
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    points, _ = detector(image)
    if points is not None:
        print("===============")
        i = 0
        j = 0
        tmp_x = [0 for p in range(21)]
        tmp_y = [0 for p in range(21)]
        for point in points:
            x, y = point
            cv2.circle(frame, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS)
            tmp_x[i] = int(x)
            tmp_y[i] = int(y)
            # point
            # print(i, " : " ,(int(x), int(y)))
            i = i + 1
        for connection in connections:
            x0, y0 = points[connection[0]]
            x1, y1 = points[connection[1]]
            cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)
            # connection
            # print(j, " : " ,(int(x0), int(y0)), " / ", (int(x1), int(y1)))
            # j = j + 1
        print("===============")
        # extract number - right hand
        number = 0
        if tmp_x[4] > tmp_x[5]:
            number = number + 1
        if tmp_y[5] > tmp_y[8]:
            number = number + 1
        if tmp_y[9] > tmp_y[12]:
            number = number + 1
        if tmp_y[13] > tmp_y[16]:
            number = number + 1
        if tmp_y[17] > tmp_y[20]:
            number = number + 1
        str1 = "Number : %d" % number
        cv2.putText(frame, str1, (100, 100), cv2.FONT_ITALIC, 1, (0, 255, 0))

        #add numbers
        temp = [0, 0, 0, 0, 0]
        p = 0
        for var in pre_num: #shift
            if p == 0:
                temp[p+1] = pre_num[p]
                p = p+1
            elif p != 4:
                temp[p+1] = pre_num[p]
                pre_num[p] = temp[p]
                p = p + 1
            else:
                pre_num[p] = temp[p]
        pre_num[0] = number
        for var in pre_num:
            print(var)
        print("=========")
        # Check
        for k in range(0, 3):
            if pre_num[k] != pre_num[k+1]:
                breaker = 0
                break
            else:
                breaker = 1
        if breaker == 1:
            sum += number

        str2 = "sum : %d" % sum
        cv2.putText(frame, str2, (100, 200), cv2.FONT_ITALIC, 1, (255, 0, 0))



    cv2.imshow(WINDOW, frame)
    hasFrame, frame = capture.read()
    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()
