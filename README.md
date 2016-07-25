# Tornado API - Clothing Segmentation with opencv3
Tornado API to segment clothing in fashion images

## Install
* Install the latest Anaconda distribtution to your system here (https://www.continuum.io/downloads) 
  * Make sure you select the correct operating system
* Install OpenCv3 - There are good online walkthroughs. See Below
  * OSX - http://www.pyimagesearch.com/2015/06/15/install-opencv-3-0-and-python-2-7-on-osx/
  * OSX - https://www.learnopencv.com/install-opencv-3-on-yosemite-osx-10-10-x/
  * Ubuntu - http://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/
* Clone this repository and create the app environment
  * `$ mkdir cs_repo && cd cs_repo`
  * `$ git clone https://github.com/ncaadam/clothing_segmentation`
  * `$ cd clothing_segementation`
  * `$ CS_HOME=$(pwd)`
  * `$ conda create -p $CS_HOME/env -y --copy anaconda`
* Make input and output folders in the repo folder (`$CS_HOME`)
  * `$ mkdir input && mkdir output`

## Dataset
* Download the test dataset here -> (TBD)
* Extract the images to the `$CS_HOME/input` folder created above

## Usage
* Navigate to the application, activate the environment, and start the API
  * `$ cd $CS_HOME`
  * `$ source activate env`
  * `$ nohup python clothing_segmentation_api.py &` (the API will run at `localhost:9999`)
* Send a POST request to the `cut` directory
  * `$ curl -v -XPOST -H 'Content-Type: application/json' -d '{"num_threads": 8}' http://localhost:9999/cut`

## Findings
* As my laptop has 4 logical cores, the API runs most efficiently at **8 or 16 threads**
  * 1 thread = 873ms/image
  * 2 threads = 453ms/image
  * 4 threads = 354ms/image  
  * 8 threads = 327ms/image
  * 16 threads = 315ms/image
  * 32 threads = 330ms/image
  * 64 threads = 332ms/image

## Future Considerations / Enhancements
* Pose estimation
* Scale invariant features
* Machine learning to identify the person
* More robust background and skin subtraction
* Add logging to a log file

## Resources and Inspirations
* Automatic Segmentation of Clothing for the Identification of Fashion Trends Using
K-Means Clustering - http://cs229.stanford.edu/proj2009/McDanielsWorsley.pdf
* Getting the Look: Clothing Recognition and Segmentation
for Automatic Product Suggestions in Everyday Photos - http://image.ntua.gr/iva/files/kalantidis_icmr13.pdf
* 2D Articulated Human Pose Estimation and Retrieval in (Almost) Unconstrained Still Images - https://www.robots.ox.ac.uk/~vgg/publications/2012/Eichner12/eichner12.pdf
