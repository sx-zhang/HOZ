# Hierarchical Object-to-Zone Graph for Object Navigation
Sixian Zhang, Xinhang Song, Yubing Bai, Weijie Li, Yakui Chu, Shuqiang Jiang (Accepted by ICCV 2021)

[ICCV 2021 Paper]() | 
[Arxiv Paper](https://arxiv.org/abs/2109.02066) | 
[Video demo](https://drive.google.com/file/d/1UtTcFRhFZLkqgalKom6_9GpQmsJfXAZC/view)
## Setup
- Clone the repository `git clone https://github.com/sx-zhang/HOZ.git` and move into the top-level directory `cd HOZ`
- Install the dependencies. `pip install -r requirements.txt`
- We provide pre-trained model of [hoz](https://drive.google.com/file/d/11L-ejoWgLHPBe_F-gQ7dJ5gQZB0dzNjr/view?usp=sharing) and [hoztpn](). For evaluation and fine-tuning training, you can download them to the `trained_models` directory.
- Download the [dataset](https://drive.google.com/file/d/1kvYvutjqc6SLEO65yQjo8AuU85voT5sC/view), which refers to [ECCV-VN](https://github.com/xiaobaishu0097/ECCV-VN). The offline data is discretized from [AI2THOR](https://ai2thor.allenai.org/) simulator.  
  Your `data` folder should look like this
  
  `Data 
    └── AI2thor_offline_data/
    ├── FloorPlan1
    │   ├── resnet18_featuremap.hdf5
    │   ├── graph.json
    │   ├── visible_object_map_1.5.json
    │   ├── det_feature_categories.hdf5
    │   ├── grid.json
    │   └── optimal_action.json
    ├── FloorPlan2
    └── ...`
## HOZ graph Construction
## Training and Evaluation
### Train the baseline model 
`python main.py --title Basemodel --model BaseModel --workers 12 –gpu-ids 0`
### Train our HOZ model 
`python main.py --title HOZ --model HOZ --workers 12 –gpu-ids 0`
### Evaluate our HOZ model 
`python python full_eval.py --title HOZ --model HOZ --results_json HOZ.json --gpu-ids 0`
### Evaluate our HOZ-TPN model 
`python python full_eval.py --title HOZTPN --model MetaMemoryHOZ --results_json HOZTPN.json --gpu-ids 0`
## Citing
