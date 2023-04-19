<h1 align="center"><b>SGD method for entropy error function with smoothing L0 regularization for neural networks</b></h1>


## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Training
We need to prepare a `config.yaml` file in order to run a training job. An example of `config.yaml` file can be seen at [./config.yaml](./config.yaml) in this repository.

Next, we can start the training by the below command
```bash
python train.py --config <path-to-the-config-file>
```

**Note**: All learning curve figures and tables in the paper are based on the training logs which can be seen through this [link](https://drive.google.com/file/d/1dfHEQHUZASXTRFIzcgf4Gqtav4M_u-Yg).

## Support
If you encounter any issue while using this repository, please open a new issue in the [issue panel](https://github.com/santapo/EEGML0/issues) or contact me via [email](mailto:trtuann34@gmail.com).