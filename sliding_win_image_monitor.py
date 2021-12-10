import cv2
import numpy as np 
import argparse
import time
import pymongo
from PIL import Image
from bson import Binary
import random
import string 
import gridfs
import os
import datetime;
myclient = pymongo.MongoClient("mongodb://localhost:27017/?readPreference=primary&appname=MongoDB%20Compass&ssl=false")
mydb = myclient["DB"]
mycol = mydb["ObjDet"]

def loadFiles():
    with open('application.properties') as file:
        file_info = dict()
        lines = file.readlines()
        for line in lines:
            infos = line.split('=')
            file_info[infos[0].strip()] = infos[1].strip()
        return file_info
    
def load_yolo():
    file = loadFiles()
    net = cv2.dnn.readNet(file["weights"], file["cfg_file"])
    classes = []
    with open(file["classes_file"], "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layers_names = net.getLayerNames()
    output_layers = [layers_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes)+10, 3))
    return net, classes, colors, output_layers , int(file["size"]), int(file["finalsize_X"]),int(file["finalsize_Y"]),int(file["slicing_size"]),int(file["offset"]),int(file["threshold"]) 

def predict_on_img_chip(img,net,size):
    h,w,c = img.shape       #h,w,c are the original height width and no. of channels of the original image. By doing this we get back our original image
    # print(img.shape)
    img = cv2.resize(img, (w,h), fx=0.4, fy=0.4)

    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (size, size), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > threshold/100:
                # Object detected
                # print(class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    #indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    #print(indexes)
    #print(boxes)
    #print(confidences)

    prediciton_values = {'boxes':boxes, 'confidences':confidences, 'class_ids':class_ids}
    return prediciton_values
def imageSizer(img,length,breadth):
    leng = img.shape[0]
    width = img.shape[1]
    if(leng>=width):   
        result = cv2.resize(img, (int(width*length/leng),length))
    else:
        result = cv2.resize(img, (breadth,int(leng*breadth/width)))
    return result
def boxMerger(box,threshold):
    temp=[]
    i=0
    while i<len(box)-1:
        mrk=1
        j=i+1
        while(i<j and j<len(box)):
            lp,arr=lap(box[i],box[j])
            if(lp):
                mrk=0
                box[i]=arr
                box.pop(j)
            else:
                j+=1
        
        if(mrk):
            i+=1
    return box
def lap(arr,arr1):
    arrFin=[]
    temp=[]
    if((arr[0]<=arr1[0] and arr1[0]<=arr[0]+arr[2] and arr[1]<=arr1[1]and arr1[1]<=arr[1]+arr[3])or(arr[0]<=arr1[0] and arr1[0]<=arr[0]+arr[2] and arr1[1]<=arr[1]and arr[1]<=arr1[1]+arr1[3])):
        arrFin.append(min(arr[0],arr1[0]))
        arrFin.append(min(arr[1],arr1[1]))
        arrFin.append(max(arr[2]+arr[0],arr1[2]+arr1[0])-arrFin[0])
        arrFin.append(max(arr[1]+arr[3],arr1[1]+arr1[3])-arrFin[1])
    temp=arr
    arr=arr1
    arr1=temp
    if(len(arrFin)==0):
        if((arr[0]<=arr1[0] and arr1[0]<=arr[0]+arr[2] and arr[1]<=arr1[1]and arr1[1]<=arr[1]+arr[3])or(arr[0]<=arr1[0] and arr1[0]<=arr[0]+arr[2] and arr1[1]<=arr[1]and arr[1]<=arr1[1]+arr1[3])):
            arrFin.append(min(arr[0],arr1[0]))
            arrFin.append(min(arr[1],arr1[1]))
            arrFin.append(max(arr[2]+arr[0],arr1[2]+arr1[0])-arrFin[0])
            arrFin.append(max(arr[1]+arr[3],arr1[1]+arr1[3])-arrFin[1])
    if(len(arrFin)>0):
        return 1,arrFin
    else:
        return 0,arrFin
        

def image_detect(img_path,net, classes, colors, output_layers,slicing_size,final_size_X,final_size_Y,size,offset,threshold):
    start_h, start_w = 0, 0
    clsBox={0:[],1:[]}
    img = cv2.imread(img_path)
    orig_img = imageSizer(img,final_size_X,final_size_Y).copy()
    img=orig_img.copy()
#     print('Image is imported.')
    width,height,channels = img.shape

#     print(img.shape)
    wNo=100*(width-slicing_size)//(offset*slicing_size)
    hNo=100*(height-slicing_size)//(offset*slicing_size)
    
#     print(wNo,hNo)
    img = cv2.resize(img, (int(slicing_size*(1+(hNo*offset/100))),int(slicing_size*(1+(wNo*offset/100)))), fx=0.4, fy=0.4)
    w_new, h_new, c_new = img.shape
#     print(img.shape)

  

    boxes = []
    confidences = []
    class_ids = []
    tiles = []



    # Breaking image into chips of size = slicing_size and then doing predictions on them
    # All the predictions are stored in boxes, confidences, class_ids, tiles
    i=0
    noUp=0
    j=0
    for i in range(0,slicing_size*offset*hNo//100,slicing_size*offset//100):
        for j in range(0,slicing_size*offset*wNo//100,slicing_size*offset//100):
#             print(str(noUp)+'/'+str(hNo*wNo)) # this is like loading bar
            noUp+=1
            # print(i,i+slicing_size,j,j+slicing_size)
            out = img[j:j+slicing_size,i:i+slicing_size,:]
            prediction_values = predict_on_img_chip(out,net,size)
            # print(prediction_values)
            chip_boxes = prediction_values['boxes']
            chip_confidences = prediction_values['confidences']
            chip_class_ids = prediction_values['class_ids']
            for ind in range(len(chip_boxes)):
                x, y, w, h = chip_boxes[ind]
                chip_boxes[ind] = x + i, y + j, w, h
                if(chip_confidences[ind]>threshold/100):
                    clsBox[chip_class_ids[ind]].append(chip_boxes[ind])
               
             
           
            



#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3) #0.4, 0.3
#     print(indexes)


    font = cv2.FONT_HERSHEY_PLAIN

    # In the loop we are putting the predicted bounding boxes on the image with proper color
    # Red is for Radome and Blue is for MeshAntenna
    labls=[]
    imgs=[]
    imgnw=orig_img.copy()
    for i in clsBox.keys():
        clsBox[i]=boxMerger(clsBox[i],1)
        for j in clsBox[i]:
            # print(i,j)
            x, y, w, h = j
            x = int(x/h_new * height)
            y = int(y/w_new * width)
            w = int(w/h_new * height)
            h = int(h/w_new * width)
            label = str(classes[i])
            color = colors[i]
            boxes.append(j)
            class_ids.append(i)
            labls.append(label)
            newimg=imgnw[y:y+h,x:x+w].copy()  
            imgs.append(newimg)
            cv2.rectangle(orig_img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(orig_img, label, (x, y-10), font, 2, color, 2)
        
#     for i in range(len(boxes)):
#          if i in indexes:
#             x, y, w, h = boxes[i]
#             x = int(x/h_new * height)
#             y = int(y/w_new * width)
#             w = int(w/h_new * height)
#             h = int(h/w_new * width)
#             label = str(classes[class_ids[i]])
#             color = colors[class_ids[i]]
#             labls.append(label)
#             newimg=imgnw[y:y+h,x:x+w].copy()  
#             imgs.append(newimg)
#             cv2.rectangle(orig_img, (x, y), (x + w, y + h), color, 2)
#             cv2.putText(orig_img, label, (x, y-10), font, 2, color, 2)
    return orig_img,imgnw, boxes,class_ids,imgs,labls

net, classes, colors, output_layers, size, final_size_X, final_size_Y, slicing_size, offset, threshold = load_yolo()

while(True):
    time.sleep(2)
    entries = os.listdir('./images/')
    if(len(entries)):
        time.sleep(5)
        for file in entries:
            print(file)
            img,imgog, boxes,class_ids,imgs,labls = image_detect("./images/"+file, net, classes, colors, output_layers,slicing_size,final_size_X,final_size_Y,size,offset,threshold)
            box=[]
            for i in range(len(boxes)):
                box.append({"dim":list(boxes[i]),"class|":int(class_ids[i])})

            ran = ''.join(random.choices(string.ascii_uppercase + string.digits, k = 5))    

            ran+=str(mycol.count())

            filename = "img11.jpg"
            cv2.imwrite(filename,imgog)
            datafile = open(filename,'rb');
            thedata = datafile.read()

            fs = gridfs.GridFS(mydb)

            fs.put(thedata, filename=ran+"org.jpg")

            cv2.imwrite(filename,img)
            datafile = open(filename,'rb');
            thedata = datafile.read()
            fs.put(thedata, filename=ran+"mpl.jpg")
            imgno=0
            subimgData=[]
            for igs in imgs:
                cv2.imwrite(filename,igs)
                datafile = open(filename,'rb');
                thedata = datafile.read()
                fs.put(thedata, filename=ran+"sub"+str(imgno)+".jpg")
                subimgData.append({"label":labls[imgno],"data":ran+"sub"+str(imgno)+".jpg"})
                imgno+=1

            

            data={"timestamp":datetime.datetime.now(),"image":ran+"org.jpg","ODAC":ran+"mpl.jpg","boxes":box,"subImagedata":subimgData}

            mycol.insert_one(data)
            os.remove("./images/"+file)
            print("./images/"+file+ " done")