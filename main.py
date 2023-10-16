import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import*
import cvzone

model = YOLO('yolov8s.pt')

def RGB(event, x, y):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x,y]
        print(colorsBGR)
        
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('D:\python\counting_enter_exit\counter_people_enter_and_exit_FST\Masuk Fakultas Saintek-1.mp4')

my_file = open('D:\python\counting_enter_exit\counter_people_enter_and_exit_FST\coco.txt')
data=my_file.read()
class_list = data.split("\n")

count=0

tracker=Tracker()

area2 = [(567,570),
         (860,580),
         (855,600),
         (542,585)]
area1 = [(527,600), #pojok kiri atas
         (843,615), #pojok kanan atas
         (838,635), #pojok kanan bawah
         (507,615)] #pojok kiri bawah

people_enter={}
counter_enter=[]

people_exit = {}
counter_exit = []


out = cv2.VideoWriter('D:\python\counting_enter_exit\output.mp4', -1, 10.0, (1280,720))
while True:
    ret,frame=cap.read()
    if not ret:
        break
    count+=1
    if count %3 != 0:
        continue
    frame=cv2.resize(frame,(1280,720))
    
    results=model.predict(frame)
    
    a=results[0].boxes.boxes
    px=pd.DataFrame(a).astype("float")
    
    list=[]
    for index, row in px.iterrows():
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'person' in c:
            list.append([x1,y1,x2,y2])
    bbox_id=tracker.update(list)
    for bbox in bbox_id:
        x3,y3,x4,y4,id=bbox
        results=cv2.pointPolygonTest(np.array(area1,np.int32),((x4,y4)),False)
        if results>0:
            people_exit[id]=(x4,y4)
        if id in people_exit:
            results1=cv2.pointPolygonTest(np.array(area2,np.int32),((x4,y4)),False)
            if results1>=0:
                cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,255),1)
                cv2.circle(frame,(x4,y4),4,(255,0,0),-1)
                cvzone.putTextRect(frame,f'{id}',(x3,y3),1,2)
                if counter_exit.count(id)==0:
                    counter_exit.append(id)
        results2=cv2.pointPolygonTest(np.array(area2,np.int32),((x4,y4)),False)
        if results2>0:
            people_enter[id]=(x4,y4)
        if id in people_enter:
            results3=cv2.pointPolygonTest(np.array(area1,np.int32),((x4,y4)),False)
            if results3>=0:
                cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,255),1)
                cv2.circle(frame,(x4,y4),4,(255,0,0),-1)
                cvzone.putTextRect(frame,f'{id}',(x3,y3),1,2)
                if counter_enter.count(id)==0:
                    counter_enter.append(id)
        
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,0,255),1)
    cv2.polylines(frame,[np.array(area2,np.int32)],True,(0,255,0),1)  
    masuk = len(counter_enter)
    keluar = len(counter_exit) 
    cvzone.putTextRect(frame,f'Mahasiswa masuk fst: {masuk}',(10,20),1,2,(255,255,255),(85,26,139))
    cvzone.putTextRect(frame,f'Mahasiswa keluar fst: {keluar}',(10,60),1,2,(255,255,255),(85,26,139))
    cvzone.putTextRect(frame,'SYSTEM PENGHITUNG MAHASISWA MASUK FST',(300,150),1,2,(205,0,0),(22, 117, 49),cv2.FONT_HERSHEY_TRIPLEX)
    cvzone.putTextRect(frame,'Di Program oleh: Ahmad Ali Sidqi A',(950,700),1,1,(0,0,255),(85,26,139))
    out.write(frame)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()