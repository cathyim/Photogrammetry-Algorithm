# A Photogrammetry Algorithm

## To run this algorithm you need:
  1) Python version +2.7
  2) numpy
  3) glob
  4) openCV2
  5) PyntCloud
  6) pandas
  
## Input
There are two methods of photo-taking. The user could station the object on a rotatable surface, such as a tray or turntable, with the camera in a fixed position and capture a photo of the object every few degrees. The user could also have the object stationed in a singular position and have the camera rotate around the object from a fixed distance from the object. For both methods, each sequential photo should be the same number of degrees apart. 

Keep in mind that algorithm is set up so that the quantity of input photos effects run-time and quality of the output. The more photos the user provides, the more accurate the 3D model will be and the longer the algorithm will run. From experimentation, I would recommend an input size of somewhere between 18-36 images. In the case of an 18 photo entry, rotate the object 20 degrees between sequential each photo, and in the case of a 36 photo entry, rotate the object 10 degrees between sequential each photo.

## Ouput
The final 3D model is generated in the form of a point cloud. For this, I used the PyntCloud method that is provided by the pandas library. The resulting product is a .PLY file that can be imported into various 3D modeling software such as Blender or Maya.

## How to Run
1) Gather all of your input photos in a folder.
2) Gather all of your calibration photos (your checkerboard photos) in another folder. 
3) Change line 4 in the main.py file to match the file path that you created for step 1.
3) Change line 6 in the main.py file to match the number of photos that are in that file.
4) Change line 100 in the cameracalibration.py file to match the folder path of you created for step 2.
5) Change line 101 in the cameracalibration.py file to the focal length = 3.3 of your camera (provided by the manufacturer).
6) Change line 102 in the cameracalibration.py file to match number of photos of your object.
7) Change line 103 in the cameracalibration.py file to match the distance from the camera and object along the floor.
8) Change line 104 in the cameracalibration.py file to match the change to height of camera.
9) Run cameracalibration.py file
10) Go to line 58 in imagematching.py and change to reflect your camera's sensor dimensions (provided by manufacturer).
11) Now, run the main method!
