
import cv2
import numpy as np
img=cv2.imread("images.jpg")
img_width=img.shape[1]
img_height=img.shape[0]
img_blob=cv2.dnn.blobFromImage(img,1/255,(416,416),swapRB=True,crop=False)
labels=["raccoon"]


colors=["100,0,150"]
colors=[np.array(color.split(",")).astype("int") for color in colors] 
colors=np.array(colors)
colors=np.tile(colors,(18,1))

model=cv2.dnn.readNetFromDarknet("yolov4_class1filter18.cfg",
                                 "yolov4_class1filter18_last.weights") #değiştir
layers=model.getLayerNames()
output_layer=[layers[layer[0]-1] for layer in model.getUnconnectedOutLayers()]
model.setInput(img_blob)
detection_layers=model.forward(output_layer)


ids_list=[]
boxes_list=[]
confidences_list=[]



for detection_layer in detection_layers:
    for object_detection in detection_layer:
        scores=object_detection[5:]
        predicted_id=np.argmax(scores)
        confidences=scores[predicted_id]
        if confidences >0.2:
            label=labels[predicted_id]
            boundingbox=object_detection[0:4]*np.array([img_width,
                                                            img_height,img_width,
                                                            img_height])
            
            (box_center_x,box_center_y,box_width,box_height)=boundingbox.astype("int")
            start_x=int(box_center_x-(box_width/2))
            start_y=int(box_center_y-(box_height/2))
           
            
            
            ids_list.append(predicted_id)
            confidences_list.append(float(confidences))
            boxes_list.append([start_x,start_y,int(box_width),int(box_height)])
            
            
max_ids=cv2.dnn.NMSBoxes(boxes_list,confidences_list,0.5,0.4)
for max_id in max_ids:
    max_class_id=max_id[0]
    box=boxes_list[max_class_id]
    start_x=box[0]
    start_y=box[1]
    box_width=box[2]
    box_height=box[3]

    predicted_id=ids_list[max_class_id]
    label=labels[predicted_id]
    confidence=confidences_list[max_class_id]
    
    end_x=start_x+box_width
    end_y=start_y+box_height
            
    box_color=colors[predicted_id]
    box_color=[int(each)for each in box_color]
            
            
            
    label="{}:{:.2f}%".format(label,confidence*100)

    print("predicted object {}".format(label))
            
    cv2.rectangle(img,(start_x,start_y),(end_x,end_y),box_color,2)  
    cv2.putText(img,label,(start_x,start_y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,box_color,2)
            
cv2.imshow("people",img)
cv2.waitKey(0)
            
            