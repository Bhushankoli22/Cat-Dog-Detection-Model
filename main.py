import torch
import cv2
import warnings
warnings.filterwarnings("ignore")

PATH = "best.pt" #model file path
model = torch.hub.load("ultralytics/yolov5","custom", PATH) #load the model using torch.hub.load() function 

# print(model,"model")
model.conf = 0.1

input_image = cv2.imread("images/train/Cat_ (6).jpg")[:,:,::-1] # read the image from disk , save it in a variable 
# dog cha path pan deu shakto "D:\Cat _Dog_detection\images\train\Dog_ (19).jpg"
result = model(input_image) 
#predict on an image
# result = model("images/test/*.jpg") #purna directory diliye
result.print() #to print results
result.crop() #to save the crop using bounding box
result.save() #to save the results

df = result.pandas().xyxy[0]
print(type(df))
df.to_csv("result.csv")
print(df)