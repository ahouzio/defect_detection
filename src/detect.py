import glob
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from tqdm import tqdm
class_explain = {"no default":1, "small residue":20, "large residue":21, "C residue":22,
                 "deviation":32, "scratch" : 41,  "small missing cupper":44,
                 "large missing cupper":45}
label_to_index = {1:0, 20:1, 21:2, 22:3, 32:4, 41:5, 44:6, 45:7}

class detection_model():
    """

    detection_model is a simple class that can be used to perform object detection on images.

    Attributes:
        model (torch.nn.Module): The pre-trained object detection model
        conf (float): The confidence threshold for object detection
        iou (float): The IoU threshold for object detection
        max_det (int): The maximum number of objects to detect per image
        amp (bool): Whether or not to use automatic mixed precision

    """
        
    def __init__(self, model_path, conf = 0.25, iou = 0.45, max_det = 1, amp = False):
        """
        The constructor for detection_model class.

        Args:
            model_path (str): The path to the pre-trained model
            conf (float): The confidence threshold for object detection (default 0.25)
            iou (float): The IoU threshold for object detection (default 0.45)
            max_det (int): The maximum number of objects to detect per image (default 1)
            amp (bool): Whether or not to use automatic mixed precision (default False)

        """  
        if torch.cuda.is_available():
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, device = 'cuda:0')
        else:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, device = 'cpu')
        
        self.model.conf = conf
        self.model.iou = iou
        self.model.max_det = max_det
        self.model.amp = amp


    def predict(self,img_path, size = 256):
        """
        This method predicts the most probable class of the image, as well as its bounding box and confidence score.

        Args:
            img_path (str): The path to the image
            size (int): The size of the image to be used for inference (default 256). Keep in mind that the size used to train the custom model is 256.

        Returns:
            list: A list containing the following elements:
                - xmin (float):  The x-coordinate of the top left corner of the bounding box
                - ymin (float):  The y-coordinate of the top left corner of the bounding box
                - xmax (float):  The x-coordinate of the bottom right corner of the bounding box
                - ymax (float):  The y-coordinate of the bottom right corner of the bounding box
                - class_label (int): The class label of the predicted object
                - confidence (float): The probability of the predicted object being in the image

        """ 
        prediction = self.model(img_path, size=size)
        res = prediction.pandas().xyxy
        
        if res[0].empty: # if no object is detected
            class_label = 1 # no defect
            return [0,0,0,0,class_label,1]
        else :
            xmin,ymin,xmax,ymax = res[0]["xmin"].item(),res[0]["ymin"].item(),res[0]["xmax"][0].item(),res[0]["ymax"].item() # take first and most probable object
            confidence = res[0]["confidence"].item()
            class_label = class_explain[res[0]["name"].item()]   
            return [xmin,ymin,xmax,ymax,class_label,confidence]
    
    def evaluate(self, img_dir, save = False, size = 256):
        """
        This method evaluate a dataset of images and returns metrics(weighted accuracy and classification ratio). Saves a confusion matrix in the results fodler.

        Args:
            img_dir (str): The path to the directory containing the images, should be in the form imglabel_idx.jpg
            size (int): The size of the image to be used for inference (default 256). Keep in mind that the size used to train the custom model is 256.
            save (bool): Whether or not to save the confusion matrix (default False)
        Returns:
            class_ratio (float): the mean of confidence_level (which must lie between 0 and 1 for each image)
            weighted_acc (float): the mean of [correctly_classified * confidence_level] / classification ratio
    
        """
        img_files = glob.glob(img_dir + "/*.jpg")
        # extract labels from image names
        labels =  [int(img_file.split("\\")[-1].split("_")[0]) for img_file in img_files]
        n = len(img_files)
        predictions = []
        weighted_acc, class_ratio = 0,0
        for i in tqdm(range(n)):
            prediction = self.predict(img_files[i], size)
            predictions.append(prediction)
            class_ratio += prediction[5]
            if prediction[4] == labels[i]:
                weighted_acc += prediction[5]
        
        weighted_acc /= class_ratio
        class_ratio /= n
        # use the predictions to create a confusion matrix
        confusion_matrix = torch.zeros(8, 8, dtype=torch.int32)
        for i in range(n):
            gt = label_to_index[labels[i]]
            pred = label_to_index[predictions[i][4]]
            confusion_matrix[gt, pred] += 1
        # save the confusion matrix as heatmap
        fig, ax = plt.subplots(figsize=(10,10))
        ax = sns.heatmap(confusion_matrix, annot=True, cmap="Blues", xticklabels=list(class_explain.keys()), yticklabels=list(class_explain.keys()), fmt = "d")
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Predicted")
        ax.set_title("Confusion Matrix")
        if save == True:
            dir_save = "..\\results\\"
            # save the confusion matrix in dir_save dir
            path = os.path.join(dir_save, img_dir.split("/")[-1] + "_confusion_matrix.png")
            print("Saving confusion matrix to {}".format(path))
            plt.savefig(path, bbox_inches="tight")
        
        return class_ratio, weighted_acc
     
        
        
        
if __name__ == '__main__':
    # specify the mode in the command line either detect or evaluate
    parser = argparse.ArgumentParser(description="This program detects defects in images or evaluates a dataset of images of defects")
    parser.add_argument("--mode", help="specify the mode of the program either detect or eval",default="detect")
    parser.add_argument("--model_path", help="specify the path to the model weights", default="../artifacts/defect_5-v0/best.pt")
    parser.add_argument("--path", help="specify the path to the image in case of detect mode", default="../test_images/0.jpg")
    parser.add_argument("--dir", help="specify the path to the directory containing the images in case of eval mode", default="../test_images")
    parser.add_argument("--save", help="specify whether or not to save the confusion matrix", default=True, type=bool)
    args = parser.parse_args()
    save = args.save
    mode = args.mode
    model_path = args.model_path
    img_path = args.path
    img_dir = args.dir
    # if its detect prompt him with the image path, if its evaluate prompt him with the image directory 
    if mode == "detect":
        model = detection_model(model_path)
        prediction = model.predict(img_path)
        print("The predicted class is: ", prediction[4], ",with a confidence of: ", prediction[5])
        print("The bounding box is: ", prediction[:4])
    elif mode == "eval":
        model = detection_model(model_path, conf = 0.25)
        class_ratio, weighted_acc = model.evaluate(img_dir, save = save)
        print("class_ratio: ", class_ratio)
        print("weighted_acc: ", weighted_acc)