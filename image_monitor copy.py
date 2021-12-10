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
    return net, classes, colors, output_layers , int(file["size"]), int(file["finalsize_X"]),int(file["finalsize_Y"],int(file["slicing_size"])) 

def predict_on_img_chip(img,net):
    # Detecting objects
    img = cv2.imread(img)
    h,w,c = img.shape       #h,w,c are the original height width and no. of channels of the original image. By doing this we get back our original image
    img = cv2.resize(img, (w,h), fx=0.4, fy=0.4)

    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

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
            if confidence > 0.3:
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
def image_detect(img_path,net, classes, colors, output_layers,slicing_size,final_size_X,final_size_Y):
    start_h, start_w = 0, 0

    t0=time()
    #img = cv2.imread(os.path.join(img_path,img_name))
    img = cv2.imread(img_path)
    orig_img = img.copy()
    print('Image is imported.')
    width,height,channels = img.shape

    print(img.shape)

    if img.shape[0] <=slicing_size and img.shape[0] >= 416:
        slicing_size = slicing_size//2
    elif img.shape[0] >= slicing_size:
        img = cv2.resize(img, (slicing_size*(height//slicing_size),slicing_size*(width//slicing_size)), fx=0.4, fy=0.4)
        w_new, h_new, c_new = img.shape
        print(img.shape)


    t0 = time()

    tiles_row = height//slicing_size
    tiles_col = width//slicing_size

    out = list(np.zeros((width//slicing_size,height//slicing_size)))
    out = [list(i) for i in out]

    boxes = []
    confidences = []
    class_ids = []
    tiles = []



    # Breaking image into chips of size = slicing_size and then doing predictions on them
    # All the predictions are stored in boxes, confidences, class_ids, tiles
    i=0
    j=0
    for i in range(tiles_col-start_w):
        for j in range(tiles_row-start_h):
            #print(i,j)
            out[i][j] = img[i*slicing_size+start_w:i*slicing_size+slicing_size+start_w,j*slicing_size+start_h:j*slicing_size+slicing_size+start_h,:]
            prediction_values = predict_on_img_chip(out[i][j],net)

            chip_boxes = prediction_values['boxes']
            chip_confidences = prediction_values['confidences']
            chip_class_ids = prediction_values['class_ids']
            for ind in range(len(chip_boxes)):
                x, y, w, h = chip_boxes[ind]
                chip_boxes[ind] = x + j*slicing_size, y + i*slicing_size, w, h
                boxes.append(chip_boxes[ind])
                confidences.append(chip_confidences[ind])
                class_ids.append(chip_class_ids[ind])
                tiles.append((tiles_col,tiles_row))
            if (i*tiles_row+j)%64==0:
                if (i*tiles_row+j)==0:
                    pass
            



    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3) #0.4, 0.3



    font = cv2.FONT_HERSHEY_PLAIN

    # In the loop we are putting the predicted bounding boxes on the image with proper color
    # Red is for Radome and Blue is for MeshAntenna
    labls=[]
    imgs=[]
    imgnw=orig_img.copy()
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            x = int(x/h_new * height)
            y = int(y/w_new * width)
            w = int(w/h_new * height)
            h = int(h/w_new * width)
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            labls.append(label)
            newimg=imgnw[y:y+h,x:x+w].copy()  
            imgs.append(newimg)
            cv2.rectangle(orig_img, (x, y), (x + w, y + h), color, 10)
            cv2.putText(orig_img, label, (x, y + 30), font, 3, color, 5)
	orig_img=imageSizer(orig_img,final_size_X,final_size_Y)
    imgnw=imageSizer(imgnw,final_size_X,final_size_Y)
    return orig_img,imgnw, boxes,class_ids,imgs,labls

net, classes, colors, output_layers, size, final_size_X, final_size_Y, slicing_size = load_yolo()
while(True):
    time.sleep(2)
    entries = os.listdir('./images/')
    if(len(entries)):
        time.sleep(5)
        for file in entries:
            print(file)
            img,imgog, boxes,class_ids,imgs,labls = image_detect("./images/"+file, net, classes, colors, output_layers,slicing_size,final_size_X,final_size_Y)
            box=[]
            for i in range(len(boxes)):
                box.append({"dim":list(boxes[i]),"class":int(class_ids[i])})

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