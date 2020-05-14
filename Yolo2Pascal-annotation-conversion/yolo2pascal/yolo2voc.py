# these files that I clone from https://github.com/hai-h-nguyen/Yolo2Pascal-annotation-conversion after fixing some bugs

from pascal_voc_io import XML_EXT
from pascal_voc_io import PascalVocWriter
from yolo_io import YoloReader
import os.path
import sys
import cv2





imgFolderPath = sys.argv[1]

# Search all yolo annotation (txt files) in this folder
for file in os.listdir(imgFolderPath):
    if file.endswith(".txt") and file != "classes.txt":
        print("Convert", file)

        annotation_no_txt = os.path.splitext(file)[0]

        imagePath = imgFolderPath + "/" + annotation_no_txt + ".jpg"
        #print("imagePath", imagePath)

        #image = QImage()
        #image.load(imagePath)
        image = cv2.imread(imagePath)
        imageShape = [image.shape[0], image.shape[1], image.shape[2]]
        #print('imageShape', imageShape)
        imgFolderName = os.path.basename(imgFolderPath)
        imgFileName = os.path.basename(imagePath)

        writer = PascalVocWriter(imgFolderName, imgFileName, imageShape, localImgPath=imagePath)


        # Read YOLO file
        txtPath = imgFolderPath + "/" + file
        tYoloParseReader = YoloReader(txtPath, image)
        shapes = tYoloParseReader.getShapes()

        #print("shapes", shapes)
        num_of_box = len(shapes)

        for i in range(num_of_box):
            label = shapes[i][0]
            xmin = shapes[i][1][0][0]
            ymin = shapes[i][1][0][1]
            x_max = shapes[i][1][2][0]
            y_max = shapes[i][1][2][1]

            writer.addBndBox(xmin, ymin, x_max, y_max, label, 0)

        writer.save(targetFile= imgFolderPath + "/" + annotation_no_txt + ".xml")
