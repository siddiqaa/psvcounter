# psvcounter
Object detection program to count relief valves on P&amp;IDs by retraining the final layer of the Google SSD Inception V2 model in Tensorflow. 

(Readme still under development)

Acknowledgments:

Thanks for Dat Tran (https://github.com/datitran) for the excellent medium article on applying transfer learning on pre-trained object detection models (https://medium.com/towards-data-science/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9)

Optional Additional Software:

- labelImg (https://github.com/tzutalin/labelImg) for bounding box annotation on training and test data
- ImageMagick (http://www.imagemagick.org/script/index.php) for manipluting bitmap (jpg) images
- GhostScript (https://www.ghostscript.com/) for pdf manipulation and conversion to jpg

Training Workflow Steps:

Conversion from PDF

1) For my use case, the images to be processed were 11 x 17 size pages in a PDF. Pages from PDF can be converted into individual jpg files using either ImageMagick or Ghostscript.

Image Segmentation into Smaller Sizes

2) The Google SSD Inception V2 model is trained on images of size 299 by 299 so I had to segment the images into smaller sections that were 122 by 129. This size gave me a good ration of full image to target bounding box. The bounding box was not too small or too large compared to the overall segment image dimensions.To handle situations where the target symbol might be split between two segments, I used the overlap feature in ImageMagick to overlapped the segmens by 20 pixels on each side.

Training Data Annotation

3) After segmenting the images, I saved them and used labelImg (https://github.com/tzutalin/labelImg) to draw bounding boxes.

Conversion of Annotation from XML to Text

4) lableImg saves the annotated data in COCO format as XML. The https://github.com/Guanghan/darknet/blob/master/scripts/voc_label.py script was used to convert the COCO format annoation to one image per line csv annotation text file.

Manual Split of Annotation Text Files into Train and Test

6) The csv file with training annotations was manually split into training and testing sets.

Integration and Conversion of Images and Annotation Files into tf_record

7) The tf_record script was used to convert the images and the csv files into tf record expected for tensorflow

Training


Classification Split

Initially, I used ImageMagick to segment each 11 x 17 page and save the resulting segments. However, the process took quite a while and I then chose to load the 11 x 17 image directly into a Numpy array and extract subsections of the array to feed into Tensorflow. This reduced the training and classification time significantly.
