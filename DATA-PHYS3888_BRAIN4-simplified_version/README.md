# DATA/PHYS3888_BRAIN_PROJECT

## GROUP MEMBERS
Fergus Ayton, Yushang Chen, Shaojie Jin, Wancheng Liu, Afrin Mathin, Pooja Satish

## Driving Question
The aim is to develop an eye-controlled E-reader, with a specific focus to produce an eye-controlled page turner for any application that can be controlled through the keyboard (e.g. books, powerpoint, gallery). Additionally, providing a more accessible and user-friendly reading experience, especially for those with disabilities and limited mobility services.

## How to run the program

### Requirements

To ensure the reproducibility, a list of packages used are listed (**NOTE:** the package versions listed below were tested to run. Different version could have different behaviors)

[Requirements](https://github.sydney.edu.au/yche7179/DATA3888_BRAIN4/blob/code_cleanup/program/requirements.txt)

After installing all the required packages, we should able to run the program by:


```shell
sudo python3 main.py
```

## Evaluation
All the evaluating images could be found under [program/Visualization](https://github.sydney.edu.au/yche7179/DATA-PHYS3888_BRAIN4/tree/code_cleanup/program/Visualization)


## Work Flow of Our Project

### Getting training data
We use several different metrics to get the signals using spiker box. We then clean the data using fft?, read the signals and store the image and signals under datasets.

[Reading training data](https://github.sydney.edu.au/yche7179/DATA3888_BRAIN4/blob/code_cleanup/program/Preparation/stream_ani.py)


### Labeling data
We found that once we have an action, the signals wave significantly. Therefore, two methods were offered to identify an action, including zero crossing and maximum value. If there is no events happening, it should have large amount of zero crossing points and maximum value could be very low.


### Making feature matrix
We use [tsfresh](https://tsfresh.readthedocs.io/en/latest/) as a tool to help generating feature matrix. We then use [Weka](https://www.weka.io/) to perfrom a BFS feature selection. 

Code for this section could be found [here](https://github.sydney.edu.au/yche7179/DATA3888_BRAIN4/blob/code_cleanup/program/Preparation/label_dataset.py)

### Classifiers
After that, we have our classifiers trained using feature matrix.

Code for this section could be found [here](https://github.sydney.edu.au/yche7179/DATA3888_BRAIN4/blob/code_cleanup/program/Preparation/Store_classifiers.py)

### Streaming condition
Lastly, we were trying to get live data and classify it using our classifiers.

[stream](https://github.sydney.edu.au/yche7179/DATA3888_BRAIN4/blob/code_cleanup/program/stream.py)

## Future work

If someone would like to develop more actions, there are the correct procedures:
- Collecting signals running [stream_ani.py](https://github.sydney.edu.au/yche7179/DATA3888_BRAIN4/blob/code_cleanup/program/Preparation/stream_ani.py)
- Labelling events and generate feature matrix running [label_dataset.py](https://github.sydney.edu.au/yche7179/DATA3888_BRAIN4/blob/code_cleanup/program/Preparation/label_dataset.py)
- Train classifiers running [Store_classifiers.py](https://github.sydney.edu.au/yche7179/DATA3888_BRAIN4/blob/code_cleanup/program/Preparation/Store_classifiers.py)
- Run the program (will need modification for streaming condition)

