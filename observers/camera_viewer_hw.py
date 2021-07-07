##
#
# Describes a simple camera viewer class that subscribes to image topics
# and shows the associated images. It also crops the depth image to the desired object. 
#
##
#
#@misc{wu2019detectron2,
#  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
#                  Wan-Yen Lo and Ross Girshick},
#  title =        {Detectron2},
#  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
#  year =         {2019}
#}
#
##

# Setup and import everything needed
from pydrake.all import *
import cv2
import torch
import tkinter as tk
from PIL import Image as Ime, ImageTk as imtk
import matplotlib

# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import Boxes


class CameraViewer(LeafSystem):
    """
    An observer which makes visualizations of camera data

                        -------------------------
                        |                       |
    color_image ------> |                       |
                        |     CameraViewer      | ------> cropped_depth_image
    depth_image ------> |                       |
                        |                       |
                        -------------------------

    """
    def __init__(self):
        LeafSystem.__init__(self)

        # Create example images which will be used to define the 
        # (abstract) import port type
        # This is currently the only change from camera_viewer.py:
        # The depth image is changed to type PixelType.kDepth16U from 32F to match the hardware input
        sample_color_image = Image[PixelType.kRgba8U](width=640,height=480)
        sample_depth_image = Image[PixelType.kDepth16U](width=641,height=480)

        # Declare input ports
        self.color_image_port = self.DeclareAbstractInputPort(
                "color_image",
                AbstractValue.Make(sample_color_image))
        self.depth_image_port = self.DeclareAbstractInputPort(
                "depth_image",
                AbstractValue.Make(sample_depth_image))

        # Declare output port
        self.DeclareAbstractOutputPort(
                "cropped_depth_image",
                lambda: AbstractValue.Make(sample_depth_image),
                self.DoCalcHere)

        # Dummy continuous variable so we update at each timestep
        self.DeclareContinuousState(1)
        
        self.once = False
        self.here = False
        self.count = 0
        self.dir = False

    def DoCalcTimeDerivatives(self, context, continuous_state):

        pass
        
        # This method is not being used anymore...

    def DoCalcHere(self, context, output):
        """
        This is called every timestep and it takes the camera image and
        uses Detectron2 to segment the input. Using the mask of the desired 
        image, the depth image is limited to just the object. This way, 
        the point cloud only represents the desired object, and the controller 
        can only find a grasp for the object.
        """
        
        # Get the images here
        color_image = self.color_image_port.Eval(context)
        depth_image = self.depth_image_port.Eval(context)

        # Only use Detectron until the point cloud is taken of the object
        if not self.here:
            image = color_image.data
            image = np.delete(image, 3, 1)

            matplotlib.image.imsave("color.jpg", image)
            #image = np.delete(depth_image.data, 3, 1)
            #matplotlib.image.imsave('dotplot.png', depth_image.data)
            im = cv2.imread("./color.jpg")
            #im = image
            #dots = cv2.imwrite('dotplot.png', depth_image.data)
            #im_flipped = cv2.rotate(im, cv2.ROTATE_180)
            #depth_flipped = cv2.rotate(dots, cv2.ROTATE_180) 
            #img = cv2.imread("./input.jpg")
            im = cv2.resize(im, dsize=(480, 270), interpolation=cv2.INTER_CUBIC)
            #depth = cv2.resize(depth_image.data, dsize=(640, 480), interpolation=cv2.INTER_CUBIC)
            
            # Use these to show the images 
            #cv2.imshow("color_image", im)
            #cv2.imshow("depth_image", depth)
            #cv2.waitKey()

            
            # Set up Detectron and use the correct model
            cfg = get_cfg()
            cfg.MODEL.DEVICE='cpu'
            

            # Different options for base models
            #cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            #cfg.merge_from_file(model_zoo.get_config_file("Misc/")
            cfg.merge_from_file(model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_R_101_FPN_1x.yaml"))
            

            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        

            # Different options for weights
            #cfg.MODEL.WEIGHTS = os.path.join('R-50.pkl') 
            #cfg.MODEL.WEIGHTS = os.path.join('model_final.pth')
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("LVISv0.5-InstanceSegmentation/mask_rcnn_R_101_FPN_1x.yaml") 
            # So far, .5 101 FPN has worked best
            
            # Different settings that can be used
            #cfg.DATALOADER.NUM_WORKERS = 2
            #cfg.SOLVER.IMS_PER_BATCH = 2
            #cfg.SOLVER.BASE_LR = 0.00025
            #cfg.SOLVER.MAX_ITER = 1000
            #cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
            #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = .5
        
            # Produce the segmented output
            predictor = DefaultPredictor(cfg)
            outputs = predictor(im)
        
            
            # We can use `Visualizer` to draw the predictions on the image.
            v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            

            # Another option to show the segmented image (uncomment both)
            cv2.imshow("image_seg",out.get_image()[:, :, ::-1])
            cv2.waitKey()
            

            # Get the masks, classes, and boxes to crop the depth image 
            boxes = outputs["instances"].pred_boxes
            boxes = boxes.tensor.numpy()

            masks = outputs["instances"].pred_masks
            masks = np.uint8(masks)

            classes = outputs["instances"].pred_classes
            print(classes)

            #depth_image.resize(640, 480)
            #masks = cv2.resize(masks, dsize=(480, 270), interpolation=cv2.INTER_CUBIC)
        
            
            def clickYes():
                self.once = True
                self.here = True
                self.count += 1
                window.destroy()

            def clickNo():
                self.count += 1
                window.destroy()
            
            n, m, p = masks.shape
            objects = 716 in classes or 992 in classes or 1036 in classes or 1131 in classes 
            #if classes contains these objects and !once:
            if (not self.once and objects) or self.dir:
                index = []
                for g in range(n):
                    if classes[g] == 716 or classes[g] == 992 or classes[g] == 1036 or classes[g] == 1131:
                        index.append(g)
                for k in range(len(index)):
                    window = tk.Tk()
                    self.dir = False
                    # Insert segmented image
                    #seg_flip = cv2.rotate(out.get_image()[:, :, ::-1], cv2.ROTATE_180)
                    cv2.imwrite("seg.jpg", out.get_image()[:, :, ::-1])
                    image_seg = Ime.open("seg.jpg")
                    test = imtk.PhotoImage(image_seg)
                    state = tk.Label(window, text="Here is what we found:")
                    state.pack()
                    statement = tk.Label(window, text="Here is what we found:", image=test)
                    statement.image = test
                    statement.place(x=0, y=0)
                    statement.pack()
                
                    # Insert mask image
                    #mask_flip = cv2.rotate(masks[k], cv2.ROTATE_180)
                    
                    for q in range(m):
                        for w in range(p):
                            if masks[index[k], q, w] != 0:
                                masks[index[k], q, w] = 100

                    cv2.imwrite("mask.jpg", masks[index[k]])
                    im_mask = Ime.open("mask.jpg")
                    test2 = imtk.PhotoImage(im_mask)
                    prompt = tk.Label(window, text="Is this the object you want to pick up?", image=test2)
                    prom = tk.Label(window, text="Is this the object you want to pick up?")
                    prompt.image = test2
                    prompt.place(x=1000, y=0)
                    prompt.pack()
                    prom.pack()

                    # Insert buttons yes/no
                    b1 = tk.Button(window, text="Yes", command = clickYes)
                    b2 = tk.Button(window, text="No", command = clickNo)
                    b1.pack()
                    b2.pack()
                
                    window.mainloop()
                    
                    if self.here: 
                        # Crop depth image based on identified mask
                        np.resize(masks, (480, 270))
                        x, y, z = depth_image.data.shape
                        for i in range(x-1):
                            for j in range(y-1):
                                if masks[index[k], i, j] == 0:
                                    depth_image.mutable_data[i, j] = 0 
                        # Save mask incase the next frame does not identify one
                        np.save('save.npy', masks)
                        break
            
            # See if there is a detected object, and crop the depth image 
            
            
            if n == 0 or not self.here:
                # Load the last mask since there were no objects detected
                masks = np.load('save.npy')
                np.resize(masks, (480, 270))
                print(masks.shape)
                x, y, z = depth_image.data.shape
                print(depth_image.data.shape)
                for i in range(x):
                    for j in range(y):
                        #if masks[0, i, j] == 0:
                        depth_image.mutable_data[i, j] = 0
            
            
            #cv2.imshow("image", depth_image.data)
            #cv2.waitKey()

            #depth_image = cv2.resize(depth_image.data, dsize=(640,480), interpolation=cv2.INTER_CUBIC) 


            # This is another method to crop the depth image using only the boxes of the objects 
            """ 
            boxes = outputs["instances"].pred_boxes
            boxes = boxes.tensor.numpy()

            #boxes = np.uint8(boxes)
            
            if len(boxes) != 0:
                x, y, x1, y1 = boxes[0]
                x = int(math.floor(x))
                y = int(math.floor(y))
                x1 = int(math.ceil(x1))
                y1 = int(math.ceil(y1))
                w = x1 - x
                h = y1 - y

                depth_image = depth_image.data[y:y+h, x:x+w]
                cv2.imshow("cropped", depth_image.data)
            """
        
        else:
            # If the point cloud of the object is already stored,
            # do not get anymore point clouds
            x, y, z = depth_image.data.shape
            for i in range(x):
                for j in range(y):
                    #if masks[0, i, j] == 0:
                    depth_image.mutable_data[i, j] = 0 

        output.SetFrom(AbstractValue.Make(depth_image))

        #print(color_image)
        #print(type(color_image))

    def isHere(self, direct=False):
        self.dir = direct
        return self.here, self.count
