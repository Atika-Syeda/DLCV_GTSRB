# DLCV_GTSRB
German Traffic Sign Recognition Benchmark Project as part of Deep Learning for Computer Vision class

## Project Structure

```
├── README.md
├── main.py
├── model.py
├── utils.py
├── evaluate.py

```

## Summary 


## Usage

### Download dataset

Please download the German Traffic Sign Recognition Benchmark (GTSRB) dataset from [here](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/publis
hed-archive.html) and extract the files in the same directory as the code.
- Training images and annotations are in “GTSRB_Final_Training_Images.zip”
- Test images are in “GTSRB_Final_Test_Images.zip” and test annotations are in “GTSRB_Final_Test_GT.zip”

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

