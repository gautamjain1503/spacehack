import ultralytics
from ultralytics import YOLO
from cv2 import rectangle, putText, FONT_HERSHEY_SIMPLEX, COLOR_BGR2GRAY , cvtColor
from typing import List


class predict:
    def __init__(self) -> None:
        pass

    def debri_pred(self,img):
        model=YOLO("best.pt")
        results = model.predict(source=img, conf=0.3, iou=0.7)
        boxes = results[0].boxes
        location=[]
        for i in boxes:
            temp=i.xyxy
            location.append([int(temp[0][0]),int(temp[0][1]), int(temp[0][2]),int(temp[0][3])])
        
        location = sorted(location)

        return location

    def draw_img(self,img,location):
        for i in location:
            img=rectangle(img, (i[0],i[1]), (i[2],i[3]), (255,0,0), thickness=2)

        return img

    def get_predicted_image(self, img):
        boxes = self.debri_pred(img)
        img=self.draw_img(img,boxes)
        print(boxes)
        return (img,len(boxes))