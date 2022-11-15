## ECG digitization

### Description
This is the source code to extract ECG signals from an image of a ECG printout.
* Input: a image of a ECG printout
* Output: extracted signal saved in a csv file

### How to extract signal from image?
* Step 1 - Perspective transform: detect four coners of ECG paper and create a new image of the 4-point perspective transformation of the raw image;
* Step 2 - Signal extraction: extract signal from warped ECG image.

### Run the code
Run the following code:
```
python ecg_digitization.py --imagepath <path to the raw image> --savepath <path to save warped image> --signalpath <path to csv file>
```
