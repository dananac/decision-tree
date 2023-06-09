Final Project Documentation

The project being tested and deployed is a Python-based project focused on machine learning using
decision trees.I used GitHub Actions to automate both the testing and the deployment of the project. In
my original Project Design Document, I stated that I planned on using Docker and Jenkins for the testing
and deployment, however I have decided to change course and use GitHub Actions for both. All of the
code is centralized on Github, and the code is built, tested, and deployed upon each new push to the main
branch. The actons automatically deploy the tested code to PyPi.

I utilized unit tests to test the following functions: building the numpy array, building a list from the array,
building a dictionary, training the data, testing the data, making a prediction for a data sample’s label,
building the decision tree, building a random forest, and testing the random forest. I used integration tests
to test how the different data storage types work together (arrays, lists, and dictionaries), how the data
storage types are passed into the decision tree training, the connection between training the data and
building the tree, how a prediction works with a trained data sample, and the connection between building
a decision tree and a random forest. The automated testing only displays the coverage for test.py, but I
have included a screenshot of the coverage report as shown in PyCharm

![README](https://user-images.githubusercontent.com/80922081/235816639-37d8b556-e686-4de5-8f50-88459d326eaf.png)
