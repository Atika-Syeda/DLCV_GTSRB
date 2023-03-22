# DLCV_GTSRB
German Traffic Sign Recognition Benchmark Project as part of Deep Learning for Computer Vision class

## Summary 


## Project setup

Use the following to clone the package:
```
git clone https://github.com/Atika-Syeda/DLCV_GTSRB.git
```
After cloning or pulling the project, the project structure will be as follows:

```
├── environment.yml
├── evaluate.py
├── main.py
├── model.py
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
├── main.py
├── model.py
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
python3 main.py --batch_size <batch_size> --epochs <epochs> --lr <learning_rate> --seed <seed> --verbose <bool> --output-dir <output_dir> 
```

### Evaluation

To evaluate the model, run the following command:

```
python3 evaluate.py
```

The file by default uses the last saved epoch for evaluation. To modify any default parameters including the model file saved in output/trained_models, use the following command:
```
python3 evaluate.py --seed <seed> --verbose <bool> --output-dir <output_dir> --model-file <model_file>
```

## References

