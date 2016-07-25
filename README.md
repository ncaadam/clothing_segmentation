# Tornado API - Clothing Segmentation with opencv3
Tornado API to segment clothing in fashion images

## Install
* Install the latest Anaconda distribtution to your system here (https://www.continuum.io/downloads) 
  * Make sure you select the correct operating system
* Install OpenCv3 - There are good online walkthroughs. See Below
  * OSX - http://www.pyimagesearch.com/2015/06/15/install-opencv-3-0-and-python-2-7-on-osx/
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
* Call a POST request to the `cut` directory
  * `$ curl -v -XPOST -H 'Content-Type: application/json' -d '{"num_threads": 8}' http://localhost:9999/cut`

## Findings
* 

## Future Considerations / Enhancements
* Pose estimation
* Scale invariant features
* Machine learning to identify the person
* More robust background and skin subtraction

## Resources and Inspirations
* Automatic Segmentation of Clothing for the Identification of Fashion Trends Using
K-Means Clustering - http://cs229.stanford.edu/proj2009/McDanielsWorsley.pdf
* Getting the Look: Clothing Recognition and Segmentation
for Automatic Product Suggestions in Everyday Photos - http://image.ntua.gr/iva/files/kalantidis_icmr13.pdf
* 2D Human Pose Estimation and Search in TV Shows and Movies - http://www.robots.ox.ac.uk/~vgg/research/pose_estimation/
