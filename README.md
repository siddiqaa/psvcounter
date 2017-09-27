# psvcounter
Object detection program to count relief valves on P&amp;IDs by retraining the final layer of the Google SSD Inception V2 model in Tensorflow. 

(Readme still under development)

Acknowledgments:

Thanks for Dat Tran (https://github.com/datitran) for the excellent medium article on applying transfer learning on pre-trained object detection models (https://medium.com/towards-data-science/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9)

Optional Additional Software:
<ul>

<li> <a href = "https://github.com/tzutalin/labelImg">labelImg</a> for bounding box annotation on training and test data</li>
<li> <a href = "http://www.imagemagick.org/script/index.php">ImageMagick</a> for manipluting bitmap (jpg) images</li>
<li> <a href = "https://www.ghostscript.com/">GhostScript</a> for pdf manipulation and conversion to jpg</li>

</ul>

<h2>Training Workflow Steps</h2>

<h3>Conversion from PDF</h3>

For my use case, the images to be processed were 11 x 17 size pages in a PDF. Pages from PDF can be converted into individual jpg files using either ImageMagick or Ghostscript.

<h3>Image Segmentation into Smaller Sizes</h3>

2) The Google SSD Inception V2 model is trained on images of size 299 by 299 so I had to segment the images into smaller sections that were 122 by 129. This size gave me a good ration of full image to target bounding box. The bounding box was not too small or too large compared to the overall segment image dimensions.To handle situations where the target symbol might be split between two segments, I used the overlap feature in ImageMagick to overlapped the segmens by 20 pixels on each side.

<h3>Training Data Annotation</h3>

3) After segmenting the images, I saved them and used labelImg (https://github.com/tzutalin/labelImg) to draw bounding boxes.

<h3>Conversion of Annotation from XML to Text</h3>

4) lableImg saves the annotated data in COCO format as XML. The https://github.com/Guanghan/darknet/blob/master/scripts/voc_label.py script was used to convert the COCO format annoation to one image per line csv annotation text file.

<h3>Manual Split of Annotation Text Files into Train and Test</h3>

6) The csv file with training annotations was manually split into training and testing sets.

<h3>Integration and Conversion of Images and Annotation Files into tf_record</h3>

7) The tf_record script was used to convert the images and the csv files into tf record expected for tensorflow

<h3>Training</h3>


Classification Split

Initially, I used ImageMagick to segment each 11 x 17 page and save the resulting segments. However, the process took quite a while and I then chose to load the 11 x 17 image directly into a Numpy array and extract subsections of the array to feed into Tensorflow. This reduced the training and classification time significantly.
