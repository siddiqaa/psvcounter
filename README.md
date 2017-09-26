# psvcounter
Object detection program to count relief valves on P&amp;IDs by retraining Resnet in Tensorflow. 

(Readme still under development)

Thanks for Dat Tran (https://github.com/datitran) for the excellent medium artile on applying transfer learning on pre-training models (https://medium.com/towards-data-science/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9)

The data was annotated using labelImg (https://github.com/tzutalin/labelImg).

Additional Steps to Dat Tran's Work

1) For my use case, the images to be processed were 11 x 17 size pages in a PDF. Pages from PDF can be converted into individual jpg files using either ImageMagick or Ghostscript.

2) The xxx model is trained on images of size xx by xx so I had to segment the images into smaller sections. This is achieved by loading images into numpy using xx library and manipulating the arrays.

3) To handle situations where the target symbol might be split between two segments, the segmentation routine overlapped the segments by 20 pixel border 

4) After segmenting the images, I saved them and used labelImg to draw bounding boxes

5) The xx script was used to convert the xml COCO format to

6) The 
