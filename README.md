# psvcounter
Tensorflow object detection example to count relief valves on P&IDs by retraining the final layer of the Google SSD Inception V2 model

Examples of relatively small symbols denoting reliev vales  to be detected <br>
<ul>
<li><img src="https://github.com/siddiqaa/psvcounter/blob/master/presentation_materials/Images/page_10%40_1214.jpg"></li>
<li><img src="https://github.com/siddiqaa/psvcounter/blob/master/presentation_materials/Images/page_19@_157.jpg"></li>
<li><img src="https://github.com/siddiqaa/psvcounter/blob/master/presentation_materials/Images/page_7@_216.jpg"></li>
</ul>

on large drawings like this

<img src="http://ptgmedia.pearsoncmg.com/images/chap1_9780132618120/elementLinks/01fig07_alt.jpg"><br>

<h2>Acknowledgments</h2>

Thanks to <a href="https://github.com/datitran">Dat Tran</a> for the excellent medium article on applying <a href="https://medium.com/towards-data-science/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9">transfer learning on pre-trained object detection models</a>

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

lableImg saves the bounding box annotation data in COCO format as XML. I then used a modified version of the <a href="https://github.com/Guanghan/darknet/blob/master/scripts/voc_label.py">conversion script</a> from <a href="https://github.com/Guanghan">Guanghan</a> to convert the individual annotation file for each trainiing image into a single CSV master file that had one line per training image containing the image file name and data on the class(es) and bounding box(es). The <a href="https://github.com/siddiqaa/psvcounter/tree/master/data">data folder</a> in my repo contains the image files, the xml annotation from labelIMG and the combined csv text file describing annotations for the all the images used for training and testing. The script for combined and convert xml annotation is <a href="https://github.com/siddiqaa/psvcounter/blob/master/data/xml_to_csv.py">xml_to_csv.py</a>. It should be run in the same directory as where all the xml files from labelIMG are stored. Line 31 should be modified to change the file name for the csv file for your own project.

<h3>Manual Split of Annotation Text Files into Train and Test</h3>

After the python script generated the csv file, I then used a spreadsheet to split the records into two files - "train_labels.csv" and "test_labels.csv". It was pure cut and paste operation. No data was edited in generating the two files. I aimed for about 10% of the records for testing. There is probably a way to automate this in Tensorflow to have random splits for each training step but I did not research and implement this for my project.

<h3>Creation of label map file</h3>

A simple json file is also needed to tell the class names for the bounding boxes in the training data. So I created the <a href="https://github.com/siddiqaa/psvcounter/blob/master/data/label_map.pbtxt">label_map.pbtxt</a> file containing the following text.<br>
```
{
  id: 1
  name: 'psv'
}
```
The id for the first object class in the map must start at 1 and not 0.

<h3>Integration and Conversion of Images and Annotation Files into tf_record</h3>

The <a href="https://github.com/siddiqaa/psvcounter/blob/master/data/generate_tfrecord.py">generate_tfrecord.py"</a> script was used to convert the images and the csv files into tf record expected for tensorflow. On Linux, the following commands were run from the data directory: <br>
```shell
python3.6 generate_tfrecord.py --csv_input=train_labels.csv  --output_path=train.record

python3.6 generate_tfrecord.py --csv_input=test_labels.csv  --output_path=test.record
```

The files train.record and test.record are created from the two commands above.

<h3>Downloading the Pre-Trained Model</h3>

The next step is to download the pre-trained model from the <a href="https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md">Tensorflow detection model zoo</a>. In this case, I used the model <a href="http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_11_06_2017.tar.gz">ssd_inception_v2_coco_11_06_2017</a> model. The tarball should be moved into the models directory and exracted with the following command.<br>
```shell
tar -xvf ssd_inception_v2_coco_11_06_2017.tar.gz
```

Finally, the <a href="https://github.com/siddiqaa/psvcounter/blob/master/models/ssd_inception_v2_coco.config">training pipeline configuration file</a> is customized. There are four customizations needed in this file:
<ol>
<li>Update the number of clases on line 9</li>
<li>Update the file location path to the train and test record files in lines 171 and 185 respectively</li>
<li>Update the file location path to the class label map created earlier in line 173 and 187.</li>
<li>Update the file location path to the pre-trained model in line 152.</li>
</ol>

<h2>Ready to Train</h2>
Finally, everything is ready to train. Training is done using scripts from the <a href="https://github.com/tensorflow/models">tensorflow models repository</a> and specifically the scripts in the <a href="https://github.com/tensorflow/models/tree/master/research/object_detection">object detection folder</a>. The repo should be cloned to your local directory and the script <a href="https://github.com/tensorflow/models/blob/master/research/setup.py">setup.py</a> in the research folder executed using pip before starting the training.<br>
```shell
python3.6 pip -m setup.py install
```

Installing the scripts as well as tensorflow on the Ubuntu machine that I used gave some unique errors that I had to research and resolve. You will likely face some errors and may have to do the same.


<to be continued ....>
