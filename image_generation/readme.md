### Requirement:
scikit-learn,
scikit-image,
numpy,
lmdb,
tqdm,
matplotlib,
pillow,
tensorflow-gpu,
scipy

### Dataset:
cifar dataset: http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
stl dataset: run download_stl.py, then run build_records.py to build tf_record file
Please remember to revise the dataset path in main.py

### Example:
Configs for many different models are provided.
For example, to reproduce the result of our framework trained without morphing, first run
```
python main.py -gpu 0 -config_file configs/fp_cifar10.yml
```
to train the model, then run 
```
python main.py -gpu 0 -config_file configs/fp_cifar10_test.yml
```
to test the trained model.

Due to randomness,  you may need to tune the hyper-parameters of sampling in the config files (current configs should be able to provide good results).

If you have any question, please contact yufanzho@buffalo.edu