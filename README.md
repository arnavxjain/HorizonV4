# HorizonV4
## A Python Integrated AI Face Recongition Engine
- The app runs mainly on two modules. `face-recognition` & `cv2`.
- In the assets folder are the images for reference acceleration. 
- After capturing data from the camera, the code compares the face in the frame to every image from `/assets`.
- The `face-recognition` module uses `linear regression` to compare the faces and gives out a result.
- Basing on that, the result is given out and is logged out on to `log.csv` file.
