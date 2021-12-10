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
    return net, classes, colors, output_layers , int(file["size"]), int(file["finalsize_X"]),int(file["finalsize_Y"]) 
net, classes, colors, output_layers, size, final_size_X, final_size_Y = load_yolo()

def load_image(img_path):
    img = cv2.imread(img_path)
    img = imageSizer(img,final_size_X,final_size_Y)
    height, width, channels = img.shape
    return img,img.copy(), height, width, channels
def detect_objects(img, net, outputLayers):
    blob = cv2.dnn.blobFromImage(img, scalefactor=0.00392, size=(size, size), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(outputLayers)
    return blob, outputs
def get_box_dimensions(outputs, height, width):
    boxes = []
    confs = []
    class_ids = []
    for output in outputs:
        for detect in output:
            scores = detect[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]
            if conf > 0.3:
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    return boxes, confs, class_ids

def draw_labels(boxes, confs, colors, class_ids, classes, img): 
    imgs=[]
    labls=[]
    imgnw=img.copy()
    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_DUPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            newimg=imgnw[y:y+h,x:x+w].copy()
            #cv2.putText(newimg, label, (5, h-5), font, 1, color, 1)
            imgs.append(newimg)
            labls.append(label)
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
    return img, boxes ,imgs, labls

def image_detect(img_path,model, classes, colors, output_layers ): 
    model, classes, colors, output_layers =model, classes, colors, output_layers 
    image,image1, height, width, channels = load_image(img_path)
    blob, outputs = detect_objects(image, model, output_layers)
    boxes, confs, class_ids = get_box_dimensions(outputs, height, width)
    img,boxes,imgs,labls = draw_labels(boxes, confs, colors, class_ids, classes, image)
#     return img,image1,boxes,class_ids
#     return imageSizer(img,final_size),imageSizer(image1,final_size), boxes,class_ids,imgs,labls
    return img,image1, boxes,class_ids,imgs,labls

def imageSizer(img,length,breadth):
    leng = img.shape[0]
    width = img.shape[1]
    if(leng>=width):   
        result = cv2.resize(img, (int(width*length/leng),length))
    else:
        result = cv2.resize(img, (breadth,int(leng*breadth/width)))
    return result

while(True):
    time.sleep(2)
    entries = os.listdir('./images/')
    if(len(entries)):
        time.sleep(5)
        for file in entries:
            print(file)
            img,imgog, boxes,class_ids,imgs,labls = image_detect("./images/"+file, net, classes, colors, output_layers)

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