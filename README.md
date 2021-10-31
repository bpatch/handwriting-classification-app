# handwriting-classification-app
Contains two streamlit apps: one to make handwritten data and one to classify based on new input (uses random forests, naive Bayes and convolutional neural networks). 

At the time of writing the app was live at http://128.199.54.172:8501/

It won't be there forever unfortunately, but these are the steps that were taken to build the app, so that it might return again:

1. Write handwriting-data-app.py
2. Open handwriting-data-app.py locally by executing "streamlit run handwriting-data-app.py" in a terminal in the working directory containing the files of this repository.
3. Training many samples for each number (I did about 10 for each number): (i) select what the number is, draw the number, click 'Add data', repeat, then (ii) click 'Save data'. 
4. Run the file 'handwriting_classification.py' to create and train the 3 models (which will be stored in 'forest', 'nb' and the 'cnn' folder. 
5. Find images of wizards on https://www.pngaaa.com/.
6. Write Dockerfile aqnd requirements.txt
7. Start a virtual machine (vm) on digitalocean (may require a vm with 16GB RAM for tensorflow). 
8. Place all files in a folder on the vm. 
9. Build the Docker image by executing "docker build -t  wizards:alpha ."
10. Run the Docker image by executing "docker run -p 8501:8501 wizards:alpha"
11. Enjoy the app!

I wrote these notes for myself, but if you are reading up to here I hope they have been useful to you too! 

