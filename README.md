# MaMMUT

This is the demo of the TMLR-2023 paper ["MaMMUT: A Simple Architecture for Joint Learning for MultiModal Tasks"](https://arxiv.org/abs/2303.16839). We plan to release the original JAX/FLAX implementation and support the model on the Cloud Vertex API, where you can train and predict with this model on Google Cloud Vertex AI Training and Prediction service. Stay tuned!

## Download the repository
You can download the repository by git clone or downloading the zip file.
```
git clone https://github.com/google-deepmind/mammut.git
```

## Installation

We use the Python built-in virtual env to set up the environment. Run the following commands:
```
PATH_TO_VENV=/path/to/your/venv
python3 -m venv ${PATH_TO_VENV}
source ${PATH_TO_VENV}/bin/activate
```

Install the requirements from the root directory.

```
pip install -r requirements.txt
```

## Download the checkpoints.
Run the following commands from the root directory.

```
cd ./checkpoints
./download.sh
```

## Run the demo.
Run the following commands from the root directory to try out the VQA demo.

```
python3 demo.py
```

## Citing this work

```latex
@article{kuo2023mammut,
  title={MaMMUT: A simple architecture for joint learning for multimodal tasks},
  author={Kuo, Weicheng and Piergiovanni, AJ and Kim, Dahun and Luo, Xiyang and Caine, Ben and Li, Wei and Ogale, Abhijit and Zhou, Luowei and Dai, Andrew and Chen, Zhifeng and others},
  journal={Transactions on Machine Learning Research},
  year={2023}
}
```

## Demo Image Source and License

Demo images come from the public VQAv2 dataset.

## License and disclaimer

Copyright 2024 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
