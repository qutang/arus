# Introduction

This demo program showcases a real-time, HAR model training, data collection and testing platform.

# Get started

## For users

Double click `arus_demo.exe` to run on Windows platform. Note that other platforms are not supported at this point.

## For developer

After installing the `arus` package, the demo program will be accessible by running

```bash
> arus_dev app run [PROJECT_ROOT] arus_demo
```

To bundle the app into executables on different platforms, run

```bash
> arus_dev app build [PROJECT_ROOT] arus_demo [VERSION_CODE]
```

and you should find a new zip file inside a folder called `releases` in the app folder.

# Testing the demo

It is recommended to test the demo program for the following tasks.

1. Train, validate, and test a HAR classifier using internal dataest.
    1. Train a custom HAR classifier on selected activities and sensor placements using the internal dataset.
    2. Validate the trained HAR classifier with LOSO validation and visualize the results in the confusion matrix.
    3. Test the trained HAR classifier using streaming data from Bluetooth sensors. There are two modes in testing. 
        1. `TEST_ONLY` will run the classifier on the streaming data and make predictions without saving any information; 
        2. `TEST_AND_SAVE` will run the classifier on the streaming data and save the raw data and annotations (when you click on the left list of the prediction panel) to disk at the same time, but no predictions nor computed features will be saved. 

2. Collect some new data in different ways.
    1. `COLLECT_ONLY`: Collect data without using any trained HAR classifier. If there is not any trained HAR classifier available, this would be the only available option. You may select the sensor placements in this case.
    2. `TEST_AND_COLLECT`: Collect data and see how the trained HAR classifier performs on the data.
    3. `ACTIVE_COLLECT`: Collect data using "active training" approach backed by the trained HAR classifier. You should be hearing generated guidance during the data collection.

3. Update the trained HAR classifier with the collected new data in different ways.
    1. `COMBINE_ORIGIN`: Combining the new data with the internal dataset to update the HAR classifier via retraining.
    2. `REPLACE_ORIGIN`: Replacing the internal dataset with the new data that have the same class labels to update the HAR classifier via retraining.
    3. `USE_NEW_ONLY`: Use only the new data to train a new HAR classifier.

4. Validate and test the updated or new trained HAR classifier.
    1. Validate the updated or new trained HAR classifier with LOSO validation and visualize the results in the confusion matrix.
    2. Test the updated or new trained HAR classifier using streaming data from Bluetooth sensors.

5. Test recovering from saved status.
    1. The current application status will be saved as a picke file on the disk. You may choose to recover from any of the saved the application status when you first start the program.

6. Saved data, logs and application status will be in a folder called `arus` in your home directory. On windows, it will be `C:\Users\[YOUR_USERNAME]\arus`.

# Feedback thoughts

Feel free to write me (tang.q@northeastern.edu) your testing feedbacks or create github [issues](https://github.com/qutang/arus/issues).

_To be added: A usability survey_

This demo program is expected to use in the [active training](https://docs.google.com/document/d/1iJdL-qI-fZProDAUjFc94gal1h11Iz6rK1BMx6zysfU/edit?usp=sharing) project, so when testing it, please keep this in mind.