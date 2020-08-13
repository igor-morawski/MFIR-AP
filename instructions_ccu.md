# Related Modules
* HTPA32x32d - toolkit for UDP communication, recording, visualization, processing TPA data, dataset-related tasks, etc.
* MFIR-AP - Mulit-view Far-InfraRed Action Prediction - development, training, testing, reporting, visualizing results, adding new labels to the dataset, also **scripts for procesing dataset!**

# Data Stages
**RAW** -> align temporally 3*TPA+RGB -> **ALIGNED** -> labeling -> **LABELED**

# Naming Conventions (filenames, filepaths)
## TPA

``YYYYMMDD_HHmm_ID{view_ID}.TXT``


## RGB

``YYYYMMDD_HHmm_IDRGB/{seconds}-{deciseconds}.jpg``

e.g. 
(...)/0-02.jpg
(...)/0-17.jpg
(...)/0-27.jpg
These will be frames taken at 0.02s, 0.12s, 0.27s, respectively. 

## Example
For a recording taken on
2020/08/05 16:07
the prefix will be 20200805_1607_
Then, we can find TPA and RGB data by adding suffixes.
For TPA, simply add ID121.TXT, ID122.TXT, ID123.TXT (recordings from view 121, 122, 123)
20200805_1607_ID121.TXT, 20200805_1607_ID122.TXT, 20200805_1607_ID123.TXT
Then, RGB frames will be in directory 20200805_1607_IDRGB (suffid IDRGB)

will be three 

# Data Format
## TPA 
Each TPA TXT file consists of:
1) File header
2) Temperature distribution

The first line of each file is always its header. Following lines, starting from line 2, are temperature values separated by space, followed by timestamp: e.g.

``2966 2942 2957 (...) t: 0.119``

This will be:
29.66 grad. C, 29.42 grad. C, 29.57 grad. C (...), all taken at time t=0.119s after initializing the recording.

## RGB
Each RGB directory consists of:
1) Frames, named ``{seconds}-{deciseconds}.jpg`` (images are timestamped in their filename)
2) timesteps.pkl 
3) timesteps.txt (optional)
4) label.txt (optional)
To load in RGB frames in a correct way use timesteps.pkl - there are two main reasons for that:
1) When globbing the file order is not be preserved. Sorting might be platform specific.
2) After alignment, some frames might be repeated for several timesteps, a sequence after alignment might look like this:
0-00.jpg, 0-14.jpg, 0-14.jpg, 0-28.jpg, (...)
If you were to load in frames without looking at timesteps.pkl you will have no way of knowing how many times one frame is repeated.
In this case timesteps.pkl will look like this:

``['0-00.jpg', '0-14.jpg', '0-14.jpg', '0-28.jpg', (...)]``

where the pickled object is a Python list of strings, so that filepath of each frame can be easily deduced:

``os.path.join({RGB_DIR_PATH},{TIMESTEP})``

e.g.:

``os.path.join("subject19","0","20200805_1607_IDRGB", "0-00.jpg"``

``os.path.join("subject19","0","20200805_1607_IDRGB", "0-14.jpg"``

``os.path.join("subject19","0","20200805_1607_IDRGB", "0-14.jpg"``

``os.path.join("subject19","0","20200805_1607_IDRGB", "0-28.jpg"``

(...)

