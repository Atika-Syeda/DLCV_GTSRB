# DLCV_GTSRB
German Traffic Sign Recognition Benchmark Project as part of Deep Learning for Computer Vision class

## Summary 
A deep neural network (DNN) architecture was implemented for a traffic sign classification task using the German Traffic Sign Recognition (GTSRB) dataset. Using a network that implemented features of  a U-Net architecture, the best performing model achieved an impressive accuracy of 98.16% on the test dataset. A class confusion matrix was used to evaluate the model and show the relative distribution of classifications for each class. The project also explored the effects of class imbalance and limited data. Using oversampling and undersampling methods, experiments were designed and run to examine how test accuracy changes when class imbalance and limited data exist. The effect was studied by comparing overall test accuracy and comparing class confusion matrices. The results demonstrated that oversampling and undersampling methods can mitigate class imbalance and improve model performance.

## Project setup

Use the following to clone the package:
```
git clone https://github.com/Atika-Syeda/DLCV_GTSRB.git
```
After cloning, the project structure will be as follows:

```
├── environment.yml
├── evaluate.py
├── main_part2.py
├── main.py
├── model.py
├── part2.ipynb
├── README.md
├── utils.py
```

Next, install anaconda and create the virtual environment as follows:
```
conda env create -f environment.yml
```
To activate the environment for running the package use:
```
conda activate GTSRB
```

## Usage

### Download dataset

Please download the German Traffic Sign Recognition Benchmark (GTSRB) dataset from [here](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html) and extract the files in the same directory as the code.
- Training images and annotations are in “GTSRB_Final_Training_Images.zip”
- Test images are in “GTSRB_Final_Test_Images.zip” and test annotations are in “GTSRB_Final_Test_GT.zip”
After downloading and extracting the dataset the project structure should look as follows:
```
├── GTSRB
    ├── Final_Test
    ├── Final_Training
    ├── GT-final_test.csv
    ├── Readme-Images-Final-Test.txt
    ├── Readme-Images.txt
├── environment.yml
├── evaluate.py
├── main_part2.py
├── main.py
├── model.py
├── part2.ipynb
├── README.md
├── utils.py
```

### Training

To train the model, run the following command:
```
python3 main.py
```
To modify any default parameters, use the following command:
```
python3 main.py --batch_size <batch_size> --epochs <epochs> --lr <learning_rate> --seed <seed> --verbose <bool> --output-dir <output_dir> --data-augmentation <data_augmentation>
```
Use --help to get mpre details for the different tags. The default command `python3 main.py` will save the output of the model/script in a folder called output by default. 

### Evaluation

To evaluate the trained model, run the following command:

```
python3 evaluate.py
```

The file by default uses the last saved epoch for evaluation. To modify any default parameters including the model file saved in output/trained_models, use the following command:
```
python3 evaluate.py --seed <seed> --verbose <bool> --output-dir <output_dir> --model-folder <model_folder>
```

This will load the model weigths from the last saved epoch to perform inference on test data. A *pred.csv* file will be saved containing the predictions for test data. A *.txt file will also be saved containing the classification accuracy. Other relevant figures will also be saved in the same output folder.

## References
Please see acknowledgements and refernce section in the attached report for details.
