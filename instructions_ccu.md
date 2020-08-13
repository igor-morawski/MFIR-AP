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

### TPA header on examples
```{subject_name},{temperature_grad_C},{humidity_%},{neg/pos},{label1_name}{label1_value},{label2_name}{label2_value},(...)```

1. Unlabeled negative
 
```subject17,23.8,49.2%,neg```

2. Labeled neg with one label
 
```subject17,23.8,49.2%,neg,label-1```

3. Labeled neg with additional custom label(s)
 
```subject17,23.8,49.2%,neg,label-1,feet-1```

4. Unlabeled positive
 
```subject17,23.8,49.2%,pos```

5. Labeled pos with one label
 
```subject17,23.8,49.2%,pos,label217```

6. Labeled pos with additional custom label(s)
 
```subject17,23.8,49.2%,pos,label217,feet226```

value>0 is considered a positive sample and value<0 (i.e. -1) is considered a negative sample.

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
ons (filenames, filepaths)
## TPA
``os.path.join("subject19","0","20200805_1607_IDRGB", "0-14.jpg"``

``os.path.join("subject19","0","20200805_1607_IDRGB", "0-14.jpg"``

``os.path.join("subject19","0","20200805_1607_IDRGB", "0-28.jpg"``

(...)

# Useful functions and classes
* HTPA32x32d.tools.
* HTPA32x32d.tools.read_txt_header
* HTPA32x32d.tools.modify_txt_header
* **HTPA32x32d.tools.txt2np** read .TXT file in as [array, timesteps]
* HTPA32x32d.tools.write_np2txt
* **HTPA32x32d.tools.apply_heatmap** apply heatmap to TPA sequence 
* HTPA32x32d.tools.timestamps2frame_durations
* **HTPA32x32d.tools.flatten_frames** 
* **HTPA32x32d.tools.reshape_flattened_frames**
* **HTPA32x32d.tools.match_timesteps** *ALIGNING DATA!*

* HTPA32x32d.dataset.TPA_Sample_from_filepaths
* HTPA32x32d.dataset.TPA_Sample_from_data
* HTPA32x32d.dataset.RGB_Sample_from_filepaths
* HTPA32x32d.dataset.TPA_RGB_Sample_from_filepaths
* HTPA32x32d.dataset.TPA_RGB_Sample_from_data

# Useful scripts
* **HTPA32x32d/recording/recorder.py** run by python -m {python_file} {arguments}
* **MFIR-AP/MFIRAP/d01_data/align.py** run locally by run by python {python_file} {arguments} *DATASET ALIGNMENT*
* **MFIR-AP/MFIRAP/d01_data/make.py** run locally by run by python {python_file} {arguments} *DATASET LABELING FROM labels.json*

# Instruction for collecting and processing data set
1. Collect data using recorder.py *HTPA32x32d/recording/recorder.py*
2. Align data set (whole directory of samples at once) *MFIR-AP/MFIRAP/d01_data/align.py*
e.g. 
python align.py filepath_header/dir_name
it will result in aligned data directory
filepath_header/dir_name+"_aligned"
3. Fill in labels in labels.json in newly created aligned data directory
4. Label data
In this step the script will add a label to each TXT file header as:

```old_header+",label{value}```

where value was read from labels.json, and 
value>0 is considered a positive sample and value<0 (i.e. -1) is considered a negative sample
e.g. 
python make.py filepath_header/dir_name+"_aligned"
it will result in aligned data directory
filepath_header/dir_name+"_labeled"

5. (Optional) If you want to add additional label use MFIRAP/labeler.py This is actually really convinient, so you might consider alternating code (writing your own scripts) to label data