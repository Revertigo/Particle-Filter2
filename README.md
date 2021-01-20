# Particle-Filter2
![particle-filter](https://user-images.githubusercontent.com/46284863/105254449-4ff4f480-5b8a-11eb-98cd-637b7879ca60.gif)  
![build-passing](https://img.shields.io/badge/build-passing-brightgreen) ![python-v.3.6.8](https://img.shields.io/badge/python-v3.6.8-blue)
# TABLE OF CONTENTS
* [Overview](#overview)

# Overview
This project is an implementation of paritcle filter algorithm as part of Navigation Algorithms course.
This time, the algorithm uses a real data in order to track and locate the moving target (Robot). The data split into four categories:  
- Map data - the (X,Y) location of the landmarks  
- Control data - velocity and yaw rate (rotation rate, angular velocity) of the moving target  
- Ground Truth - the actual postion of the moving target in (X,Y)  
- Observations from landmarks
