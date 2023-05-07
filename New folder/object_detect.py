import cv2

#open DNN
net = cv2.dnn.readNet("yolov4-tiny-custom_last.weights", "yolov4-tiny-custom.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size =(416,416), scale = .1/255)

#load classes
classes = []
with open("classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

#print("Object list")
#print(classes)

#capture video
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1366)

while True:

    #get frame
    ret, frame = cap.read()

    #object detection
    (class_ids, scores, bboxes) = model.detect(frame)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]

        #font
        font = cv2.FONT_ITALIC
        cv2.putText(frame, str(class_name), (x, y-5), font, 1, (51, 255, 51))
        
        #color(B,G,R)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (51, 255, 51), 2)

    print("class_ids", class_ids)
    print("scores", scores)
    print("bboxes", bboxes)

    cv2.imshow("Alpha Object-Detect", frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

