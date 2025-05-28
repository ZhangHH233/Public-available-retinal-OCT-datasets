眼底OCT公开数据集和预处理代码。项目持续更新中……

数据集信息汇总（Public available retinal OCT datasets）

|      Dataset  数据集      |                  Volume 数据量                  |                 Disease 病理             | Download Link 下载链接                  |
|:----------------:|:-----------------------------------------:|:--------------------------------------:|:--------------------------------------:|
|     DUKE-AMD     |                 20 volumes                |                 20 AMD                 |http://www.duke.edu/ sf59/Chiu_IOVS_2011_dataset.htm|
|    DUKE-WLOA     |               38800 B-scans               |             269 AMD; 115 H.            |http:// people.duke.edu/wsf59/RPEDC_Ophth_2013_dataset.htm|
|     DUKE-DME     |                 45 volumes                |        15 Dry AMD, 15 DME; 15 H.       |http://www.duke.edu/~sf59/Srinivasan_BOE_2014_dataset.htm.|
|     DUKE-Cyst    |                 6 volumes                 |                  6 DME                 |www.duke.edu/~sf59/Chiu_BOE_2014_dataset.htm.|
|       SPIE       |                 19 volumes                |                  19 H.                 |https://www.kaggle.com/datasets/kmader/eye-oct-datasets|
|       MIAMI      |                 50 B-scans                |               10 mild DR               | https://www.ebi.ac.uk/biostudies/europepmc/studies/S-EPMC5025289|
|      OPTIMA      |                 30 B-scans                |                   IRF                  | https://optima.meduniwien.ac.at/optima-segmentation-challenge-1/ |
|       ROCC       |                     -                     |                 DR、H.                 | ...|
|       HC-MS      |                 49 B-scans                |              14 H., 21 MS              |http://iacl.jhu.edu/Resources|
|       Cell       |              108,312 B-scans              |          H., CNV, DME, Drusen          | https://data.mendeley.com/datasets/rscbjbr9sj/3|
|      BIOMISA     | 2497 B-scans，19 C-scans，64 fundus scans |    14 AMD, 13 ME, 50 H., 26 Glaucoma   |  http://biomisa.org/index.php/glaucoma-fundus-oct-dataset/ |
|      Zenodo      |                1100 B-scans               |             847 G., 263 H.             | https://zenodo.org/records/14926793 |
|   AI-challenger  |                100 volumes                |              REA, SRF, PED             | https://github.com/AIChallenger/AI_Challenger_2018
|      RETOUCH     |                112 volumes                |              IRF, SRF, PED             | https://retouch.grandchallenge.org |
|   Isfahan MISP   |                     -                     |                    -                   | ... |
|       OCTID      |                470 B-scans                |     206 H., 102 MH, 55 AMD, 107 DR     | https://dataverse.scholarsportal.info/dataverse/OCTID |
|   RAJA-Glaucoma  |      50 OCT volumes and fundus scans      |           18 H., 32 Glaucoma           |...|
|       AROI       |                1136 B-scans               |                 24 nAMD                | https://ipg.fer.hr/ipg/resources/oct_image_database |
|     OCTA-500     |                500 volumes                |       H., AMD, DR, CNV, CSC, RVO       | https://ieee-dataport.org/open-access/octa-500|
| Retinal OCT - C8 |               24000 B-scans               | AMD、CNV、CSR、DME、DR、Drusen、MH、H. | ...|
|       GOALS      |                300 B-scans                |           53 H ., 13 Glaucoma          | https://aistudio.baidu.com/aistudio/competition/detail/230|
|    ORC SS-OCT    |                 4 volumes                 |                  4 RRD                 | https://data.mendeley.com/datasets/bzsc7gd9p3/2|
| OCTDL| 2000   B-Scans| 1231 AMD; 147 DME; 155 EM; 332 Normal; 22 RAO; 101 RVO; 76 Vitreomacular Interface Disease| https://data.mendeley.com/datasets/sncdhf53xc/4|
| HD-OCT MH| 2658 B-scans | 493 MH | https://www.kaggle.com/datasets/mathieugodbout/oct-postsurgery-visual-improvement|
| OCT5k|1672 B-scans |722 AMD, 422 DME, 530 H.| https://rdr.ucl.ac.uk/articles/dataset/OCT5k_A_dataset_of_multi-disease_and_multi-graded_annotations_for_retinal_layers/22128671/4|


注：(1)缩写：H.:健康；G.:青光眼；POAG:原发性开角型青光眼；AMD：年龄相关性黄斑病变；nAMD：新生血管性AMD；CNV：脉络膜新生血管；DR：糖尿病视网膜病变；DME：糖尿病性黄斑水肿；MH：黄斑裂孔；Drusen：玻璃膜疣；MS：多发性硬化症；
（2）补充说明：“-”表示数据提供方未提供该项参数。

预处理代码见 https://github.com/ZhangHH233/OCT_Public_dataset_preprocess/tree/main。

Notation:
（1）Abbreviations: H.: Health; G.: Glaucoma; POAG: Primary Open-Angle Glaucoma; AMD: Age-Related Macular Degeneration; nAMD: Neovascular AMD; CNV: Choroidal Neovascularization; DR: Diabetic Retinopathy; DME: Diabetic Macular Edema; MH: Macular Hole; Drusen: Drusen; MS: Multiple Sclerosis; EM: Epiretinal Membrane; RAO: Retinal Artery Occlusion; RVO: Retinal Vein Occlusione'
(2) Supplementary Explanation: “-” indicates that the data provider did not provide this parameter.

Pre-processing code in https://github.com/ZhangHH233/OCT_Public_dataset_preprocess/tree/main.

欢迎引用综述论文：
```bibtex
@article{zhang2025retinal,
  title={Retinal OCT image segmentation with deep learning: A review of advances, datasets, and evaluation metrics},
  author={Zhang, Huihong and Yang, Bing and Li, Sanqian and Zhang, Xiaoqing and Li, Xiaoling and Liu, Tianhang and Higashita, Risa and Liu, Jiang},
  journal={Computerized Medical Imaging and Graphics},
  pages={102539},
  year={2025},
  publisher={Elsevier}
} 





