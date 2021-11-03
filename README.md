# Hierarchical Object-to-Zone Graph for Object Navigation
Sixian Zhang, Xinhang Song, Yubing Bai, Weijie Li, Yakui Chu, Shuqiang Jiang (Accepted by ICCV 2021)

[ICCV 2021 Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Zhang_Hierarchical_Object-to-Zone_Graph_for_Object_Navigation_ICCV_2021_paper.html) | 
[Arxiv Paper](https://arxiv.org/abs/2109.02066) | 
[Video demo](https://drive.google.com/file/d/1UtTcFRhFZLkqgalKom6_9GpQmsJfXAZC/view)
## Setup
- Clone the repository `git clone https://github.com/sx-zhang/HOZ.git` and move into the top-level directory `cd HOZ`
- Create conda environment. `conda env create -f environment.yml`
- Activate the environment. `conda activate ng`
- We provide pre-trained model of [hoz](https://drive.google.com/file/d/11L-ejoWgLHPBe_F-gQ7dJ5gQZB0dzNjr/view?usp=sharing) and [hoztpn](https://drive.google.com/file/d/1hoqBLO6Oaty-TKT7a2slnhVx0wYi7LsC/view?usp=sharing). For evaluation and fine-tuning training, you can download them to the `trained_models` directory.
- Download the [dataset](https://drive.google.com/file/d/1kvYvutjqc6SLEO65yQjo8AuU85voT5sC/view), which refers to [ECCV-VN](https://github.com/xiaobaishu0097/ECCV-VN). The offline data is discretized from [AI2THOR](https://ai2thor.allenai.org/) simulator.  
The `data` folder should look like this
```python
  data/ 
    └── Scene_Data/
        ├── FloorPlan1/
        │   ├── resnet18_featuremap.hdf5
        │   ├── graph.json
        │   ├── visible_object_map_1.5.json
        │   ├── det_feature_categories.hdf5
        │   ├── grid.json
        │   └── optimal_action.json
        ├── FloorPlan2/
        └── ...
```
## HOZ graph Construction (Updating)
## Training and Evaluation
### Train the baseline model 
`python main.py --title Basemodel --model BaseModel --workers 12 -–gpu-ids 0`
### Train our HOZ model 
`python main.py --title HOZ --model HOZ --workers 12 -–gpu-ids 0`
### Evaluate our HOZ model 
`python full_eval.py --title HOZ --model HOZ --results-json HOZ.json --gpu-ids 0`
### Evaluate our HOZ-TPN model 
`python full_eval.py --title TPNHOZ --model MetaMemoryHOZ --results-json HOZTPN.json --gpu-ids 0`
## Citing
If you find this project useful in your research, please consider citing:
```
@InProceedings{Zhang_2021_ICCV,
    author    = {Zhang, Sixian and Song, Xinhang and Bai, Yubing and Li, Weijie and Chu, Yakui and Jiang, Shuqiang},
    title     = {Hierarchical Object-to-Zone Graph for Object Navigation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {15130-15140}
}
```
