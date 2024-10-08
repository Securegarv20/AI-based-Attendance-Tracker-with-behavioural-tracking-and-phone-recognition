﻿# AI-based Attendance Tracker with Behavioural Tracking and Phone Recognition

![Face Rec Eye Open Photo](test1.png)
![Face RecnEye Close Photo](test2.png)

## Overview
This project is designed to track attendance using AI and facial recognition. It can also monitor student behavior during class, detecting if students are distracted or not paying attention. Additionally, it includes a functionality for phone usage detection.

## Features
- Facial recognition for attendance tracking
- Behavioural tracking (eyes open/closed, distraction)
- Phone usage detection (to be implemented)

## Prerequisites
Before running the scripts, ensure you have the following software installed:
- Python 3.x
- OpenCV
- Other required libraries (see `requirements.txt`)

## Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Securegarv20/AI-based-Attendance-Tracker-with-behavioural-tracking-and-phone-recognition.git
   cd AI-based-Attendance-Tracker-with-behavioural-tracking-and-phone-recognition
2. **Install the required packages: Run the following command to install all the necessary libraries**
   ```bash
   pip install -r requirements.txt

3. **Collect face data: Run the following script to collect facial data (with and without glasses):**
   ```bash
   python image_collect_v2.py

4. **Run the main attendance logging script: Once the face data has been collected, execute:**
   ```bash
   python student_log_test_v2.py

## Log Files
Logs of each activity can be found in the log_data folder, under a subfolder named after the user, containing a log.txt file with live logs including proper text and timestamps.

## Additional Information
Make sure to have the see `haarcascade_frontalface_default.xml` file available for facial recognition. This file is required for running the detection algorithms.

## Contribution
Feel free to fork the repository, make changes, and submit a pull request if you have improvements or bug fixes.

