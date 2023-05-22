# AVT-AI-Image-Dataset and Evaluation

In this repository the data and evaluation scripts are collected to reproduce the results of the paper `goering2023ai` (see Acknowledgments).
The dataset can be also used for additional evaluation, the subjective scores for appeal, realism, and text prompt matching are included.


![](own_p23.png)
![](dalle_p23.png)

## Structure

* `images`: here for all used AI generators the generated images are stored
  * `images/prompts.csv`: the used text prompts for the generated images
* `features`: includes the scripts to calcualte the signal based features 
* `evaluation`: scripts for the evaluation, including also calculated metrics and features
    * `evaluation/all_ratings.csv`: has all mean values calculated for the subjective annotations
    * `evaluation/subjective/*`: has the raw data of all ratings

## Requirements
The evaluation scripts are only tested on linux systems (e.g. Ubuntu 20.04, 22.04).

The following software is required to reproduce the results:

* python3, python3-pip, python3-venv
* for python the following packages are required: 
    * jupyterlab
    * pandas
    * seaborn
    * cpbd
    * numpy
    * opencv-python
    * scikit-image
    * scikit-video
    * scikit-learn
* git
* the included image quality metrics have been calculated with [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch).

## Acknowledgments

If you use this software or data in your research, please include a link to the repository and reference the following papers.

```bibtex
@inproceedings{goering2023ai,
  title={Analysis of Appeal for realistic AI-generated Photos},
  author={Steve G\"oring and Rakesh {Rao Ramachandra Rao} and Rasmus Merten and Alexander Raake},
  journal={IEEE Access},
  year={2023}
}

@inproceedings{goering2023aiquality,
  author={Steve {G{\"o}ring} and Rakesh {Rao Ramachandra Rao} and Alexander Raake},
  title="Appeal and quality assessment for AI-generated images",
  booktitle="15th Int. Conference on Quality of Multimedia Experience (QoMEX)",
  year={2023},
}
```

## License
GNU General Public License v3. See [LICENSE.md](./LICENSE.md) file in this repository.
