## Weakly Supervised 3D Object Detection from Point Clouds (VS3D)
![](demo.gif)
#### Created by [Zengyi Qin](http://www.qinzy.tech/), Jinglu Wang and Yan Lu. The repository contains an implementation of this [ACM MM 2020 Paper].


### Prerequisites
- Python 3.6
- Tensorflow 1.12.0


### Quick Demo with Jupyter Notebook
Clone this repository.
```bash
git clone https://github.com/Zengyi-Qin/Weakly-Supervised-3D-Object-Detection.git
```
Enter the main folder and run installation.
```bash
pip install -r requirements.txt
```
Download the [demo data](https://drive.google.com/file/d/1s4G3avlud7H4oqEBpi0GMnL20HjPJ9Wd/view?usp=sharing) to the main folder and unzip
```bash
unzip vs3d_demo.zip
```
The demo data contains an examplary frame in the KITTI dataset and the pretrained student model. Readers can try out the quick demo with jupyter notebook.
```bash
cd core
jupyter notebook demo.ipynb
```