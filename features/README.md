# Image feature tool
The provided tool extracts some signal based features using images as input.
The features are:

* niqe
* color_fulness
* tone
* blur
* saturation
* fft
* si
* contrast
* noise
* dominant_color
* cpbd
* blur_stength


## Requirements
The tool is only tested on linux systems (e.g. Ubuntu 20.04, 22.04).

The following software is required to run:

* python3, python3-pip
* pip3 requirements: `pip3 install -r requirements.txt`

It is recommended to run this tool in a virtual environment, e.g. create with `python3 -m venv venv && source venv/bin/activate`

## Usage
```
usage: imt.py [-h] [--cpu_count CPU_COUNT] [--report_file REPORT_FILE] image [image ...]

extract several image features

positional arguments:
  image                 image to be evaluated

optional arguments:
  -h, --help            show this help message and exit
  --cpu_count CPU_COUNT
                        thread/cpu count (default: 4)
  --report_file REPORT_FILE
                        file to store predictions (default: features.json)

stg7 2023

```

## Acknowledgments

If you use this software or data in your research, please include a link to the repository and reference the following paper.

```
@inproceedings{goering2023ai,
  title={Analysis of Appeal for realistic AI-generated Photos},
  author={Steve G\"oring and Rakesh {Rao Ramachandra Rao} and Rasmus Merten and Alexander Raake},
  journal={IEEE Access},
  year={2023},
  note={to appear}
}
```
