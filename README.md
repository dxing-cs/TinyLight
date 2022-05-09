# Adaptive Traffic Signal Control on Devices with Extremely Limited Resources

Code for _Adaptive Traffic Signal Control on Devices with Extremely Limited Resources_ (IJCAI 2022). 

## Dependencies
- python=3.6.13
- torch=1.9.0+cu111
- numpy=1.19.2
- CityFlow=0.1.0 

We made some modifications based on [CityFlow v0.1.0](https://github.com/cityflow-project/CityFlow) to enrich its APIs. To ensure reproducibility, please install our adapted CityFlow by: 
```shell
cd CityFlow
pip install .
```

## Experiments on ATSC

Note: all the following steps are performed in `TinyLight` folder.  
### STEP 0: modify `config.json` (optional)

### STEP 1: run baseline 
```shell
python 01_run_baseline.py --model=FixedTime --dataset=Atlanta
```

### STEP 2: sub-graph extraction
The model will be stored in `{log_path}/TinyLight/model/`.
```shell
python 02_run_tiny_light.py --dataset=Atlanta
```

### STEP 3: post-training quantization 

This step converts the floating-point operations into integer-only ones. It will generate a series of model files in `{log_path}/TinyLightQuan/pc` and 
`{log_path}/TinyLightQuan/mcu`, which are used for evaluation in step 4. 
```shell
python 03_run_tinylight_quan.py --dataset=Atlanta 
```

### STEP 4: evaluate TinyLight (w/ PTQ)

You can run TinyLight on an Arduino Uno (with ATmega328P as the default MCU) by: 
 
- replacing `TinyLight_MCU/model.h` with the corresponding `.h` file in `{log_path}/TinyLightQuan/mcu`, and
- running the project in MCU 


If you do not have an Arduino board on hand, we also provide an equivalent implementation of TinyLight (w/ PTQ) on computers to verify its performance on traffic. For this, please: 
```shell 
cd {log_path}/TinyLightQuan/mcu/ 
python setup.py build_ext --inplace
cd {project_dir}
python 04_run_tinylight_mcu.py --dataset=Atlanta 
```

## BibTeX

If you find our work helpful in your research, please consider citing our paper:
```tex
@article{TinyLight22,
  author    = {Dong Xing and
               Qian Zheng and
               Qianhui Liu and
               Gang Pan},
  title     = {TinyLight: Adaptive Traffic Signal Control on Devices with Extremely Limited Resources},
  journal   = {CoRR},
  volume    = {abs/2205.00427},
  year      = {2022},
  url       = {https://doi.org/10.48550/arXiv.2205.00427},
  doi       = {10.48550/arXiv.2205.00427},
  eprinttype = {arXiv},
  eprint    = {2205.00427},
}
```
