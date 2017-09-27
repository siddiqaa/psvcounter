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

The Google SSD Inception V2 model is trained on images of size 299 by 299 so I had to segment the images into smaller sections that were 122 by 129. This size gave me a good ratio of segment image dimensions to target object bounding box dimension. In other words, the bounding box for the object to be detected was not too small or too large compared to the overall segment image dimensions. To handle situations where the target object to be detected might be split between two different segments, I used the overlap feature in ImageMagick to overlap the segments by 20 pixels on each side.

<h3>Training Data Annotation</h3>

After saving the image segments, I used <a href="https://github.com/tzutalin/labelImg">labelImg</a> to draw bounding boxes.

<h3>Conversion of Annotation from XML to Text</h3>

lableImg saves the bounding box annotation data in COCO format as XML. I then used a modified version of the <a href="https://github.com/Guanghan/darknet/blob/master/scripts/voc_label.py">conversion script</a> from <a href="https://github.com/Guanghan">Guanghan</a> to convert the individual annotation file for each trainiing image into a single CSV master file that had one line per training image containing the image file name and data on the class(es) and bounding box(es). The <a href="https://github.com/siddiqaa/psvcounter/tree/master/data">data folder</a> in my repo contains the image files, the xml annotation from labelIMG and the combined csv text file describing annotations for the all the images used for training and testing. The script for combined and convert xml annotation is <a href="https://github.com/siddiqaa/psvcounter/blob/master/data/xml_to_csv.py">here</a>. It should be run in the same directory as where all the xml files from labelIMG are stored. Line 31 should be modified to change the file name for the csv file for your own project.

<h3>Manual Split of Annotation Text Files into Train and Test</h3>

After the python script generated the csv file, I then used a spreadsheet to split the records into two files - "train_labels.csv" and "test_labels.csv". It was pure cut and paste operation. No data was edited in generating the two files. I aimed for about 10% of the records for testing. There is probably a way to automate this in Tensorflow to have random splits for each training step but I did not research and implement this for my project.

<h3>Creation of label map file</h3>

A simple json file is also needed to tell the class names for the bounding boxes in the training data. So I created the <a href="https://github.com/siddiqaa/psvcounter/blob/master/data/label_map.pbtxt">label_map.pbtxt</a> file containing the following text.<br>
'''JSON
{
  id: 1
  name: 'psv'
}
'''
The id for the first object class in the map must start at 1 and not 0.

<h3>Integration and Conversion of Images and Annotation Files into tf_record</h3>

The tf_record script was used to convert the images and the csv files into tf record expected for tensorflow

<h3>Training</h3>


Classification Split

Initially, I used ImageMagick to segment each 11 x 17 page and save the resulting segments. However, the process took quite a while and I then chose to load the 11 x 17 image directly into a Numpy array and extract subsections of the array to feed into Tensorflow. This reduced the training and classification time significantly.
