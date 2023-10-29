# Data Optimization in Deep Learning: A Survey
:hugs: The repository is a collection of resources on data optimization in deep learning, serving as a supplement to our survey paper "[Data Optimization in Deep Learning: A Survey](https://arxiv.org/abs/2310.16499)". If you have any recommendations for missing work and any suggestions, please feel free to [pull requests](https://github.com/YaoRujing/Data-Optimization/pulls) or [contact us](mailto:wuou@tju.edu.cn):envelope:.

If you find our survey useful, please kindly cite our paper:

```bibtex
@article{wu2023data,
  title={Data Optimization in Deep Learning: A Survey},
  author={Wu, Ou and Yao, Rujing},
  journal={arXiv preprint arXiv:2310.16499},
  year={2023}
}
```

**📝 Table of Contents**
- [Introduction](#introduction)
- [Related studies](#related-studies)
- [Overall of The proposed taxonomy](#overall-of-the-proposed-taxonomy)
- [Goals, scenarios, and data objects](#goals-scenarios-and-data-objects)
  - [Optimization goals](#optimization-goals)
  - [Application scenarios](#application-scenarios)
  - [Data objects](#data-objects)                                                            
- [Optimization pipeline](#optimization-pipeline)
  - [Data perception](#data-perception)
    - [Perception on different granularity levels](#perception-on-different-granularity-levels)                          
    - [Perception on different types](#perception-on-different-types)                           
    - [Static and dynamic perception](#static-and-dynamic-perception)                           
  - [Analysis on perceived quantities](#analysis-on-perceived-quantities)                          
  - [Optimizing](#optimizing)                           
- [Data optimization techniques](#data-optimization-techniques)
  - [Data resampling](#data-resampling)                                          
  - [Data augmentation](#data-augmentation)
    - [Augmentation target](#augmentation-target)                                         
    - [Augmentation strategy](#augmentation-strategy)                                      
  - [Data perturbation](#data-perturbation)  
    - [Perturbation target](#perturbation-target) 
    - [Perturbation direction](#perturbation-direction)                             
    - [Perturbation granularity](#perturbation-granularity)                             
    - [Assignment manner](#assignment-manner)                                                                
  - [Data weighting](#data-weighting)
    - [Weighting granularity](#weighting-granularity) 
    - [Dependent factor](#dependent-factor)                             
    - [Assignment manners](#assignment-manners)                                                 
  - [Data pruning](#data-pruning)
    - [Dataset distillation](#dataset-distillation) 
    - [Subset selection](#subset-selection)                                                                   
  - [Other typical techniques](#other-typical-techniques)   
    - [Pure mathematical optimization](#pure-mathematical-optimization) 
    - [Technique combination](#technique-combination)                                                                          
- [Data optimization theories](#data-optimization-theories)
  - [Formalization](#formalization)                                          
  - [Explanation](#explanation)
- [Connections among different techniques](#connections-among-different-techniques)
  - [Connections via data perception](#Connection-via-data-perception)                                          
  - [Connections via application scenarios](#connections-via-application-scenarios)
  - [Connections via similarity or opposition](#connections-via-similarity-or-opposition)
  - [Connections via theory](#connections-via-theory)
- [Future Directions](#future-directions)
  - [Principles of data optimization](#principles-of-data-optimization)                                          
  - [Interpretable data optimization](#interpretable-data-optimization)
  - [Human-in-the-loop data optimization](#human-in-the-loop-data-optimization)
  - [Data optimization for new challenges](#data-optimization-for-new-challenges)            
  - [Data optimization agent](#data-optimization-agent) 

# Introduction
1. **Data collection and quality challenges in deep learning: A data-centric ai perspective.**<br>
*Whang, Steven Euijong and Roh, Yuji and Song, Hwanjun and Lee, Jae-Gil.*<br>
The VLDB Journal 2023. [[Paper](https://link.springer.com/article/10.1007/s00778-022-00775-9)]
2. **The Principles of Data-Centric AI.**<br>
*Jarrahi, Mohammad Hossein and Memariani, Ali and Guha, Shion.*<br>
Communications of the ACM 2023. [[Paper](https://dl.acm.org/doi/abs/10.1145/3571724)]
3. **Multi-view graph learning by joint modeling of consistency and inconsistency.**<br>
*Liang, Youwei and Huang, Dong and Wang, Chang-Dong and Philip, S Yu.*<br>
TNNLS 2022. [[Paper](https://ieeexplore.ieee.org/abstract/document/9843949)]
4. **Latent heterogeneous graph network for incomplete multi-view learning.**<br>
*Zhu, Pengfei and Yao, Xinjie and Wang, Yu and Cao, Meng and Hui, Binyuan and Zhao, Shuai and Hu, Qinghua.*<br>
IEEE Transactions on Multimedia 2022. [[Paper](https://ieeexplore.ieee.org/abstract/document/9721669)]
5. **A close look at deep learning with small data.**<br>
*Brigato, Lorenzo and Iocchi, Luca.*<br>
ICPR 2021. [[Paper](https://ieeexplore.ieee.org/abstract/document/9412492)]
6. **Semantic Redundancies in Image-Classification Datasets: The 10% You Don't Need.**<br>
*Birodkar, Vighnesh and Mobahi, Hossein and Bengio, Samy.*<br>
arXiv 2019. [[Paper](https://arxiv.org/abs/1901.11409)]
7. **Can Data Diversity Enhance Learning Generalization?**<br>
*Yu, Yu and Khadivi, Shahram and Xu, Jia.*<br>
COLING 2022. [[Paper](https://aclanthology.org/2022.coling-1.437/)]
8. **Learning under concept drift: A review.**<br>
*Lu, Jie and Liu, Anjin and Dong, Fan and Gu, Feng and Gama, Joao and Zhang, Guangquan.*<br>
TKDE 2018. [[Paper](https://ieeexplore.ieee.org/abstract/document/8496795)]
9. **Towards a robust deep neural network against adversarial texts: A survey.**<br>
*Wang, Wenqi and Wang, Run and Wang, Lina and Wang, Zhibo and Ye, Aoshuang.*<br>
TKDE 2021. [[Paper](https://ieeexplore.ieee.org/abstract/document/9557814)]
10. **Fairness-aware classification: Criterion, convexity, and bounds.**<br>
*Wu, Yongkai and Zhang, Lu and Wu, Xintao.*<br>
arXiv 2018. [[Paper](https://arxiv.org/abs/1809.04737)]
11. **Towards a robust and trustworthy machine learning system development: An engineering perspective.**<br>
*Xiong, Pulei and Buffett, Scott and Iqbal, Shahrear and Lamontagne, Philippe and Mamun, Mohammad and Molyneaux, Heather.*<br>
Journal of Information Security and Applications 2022. [[Paper](https://www.sciencedirect.com/science/article/pii/S2214212622000138?casa_token=h4LBo3iPosgAAAAA:OQO0KoASGfPJ9_adGMCdCSGpAGyjfTiwml2_rzb9ENMQBfZRXTS3NJASUDT0Yhrl1pnFXC5A7WM)]
12. **mixup: Beyond Empirical Risk Minimization.**<br>
*Zhang, Hongyi and Cisse, Moustapha and Dauphin, Yann N and Lopez-Paz, David.*<br>
ICLR 2018. [[Paper](https://openreview.net/pdf?id=r1Ddp1-Rb)]
13. **Compensation learning.**<br>
*Yao, Rujing and Wu, Ou.*<br>
arXiv 2021. [[Paper](https://arxiv.org/abs/2107.11921)]
# Related studies
14. **Deep generative mixture model for robust imbalance classification.**<br>
*Wang, Xinyue and Jing, Liping and Lyu, Yilin and Guo, Mingzhe and Wang, Jiaqi and Liu, Huafeng and Yu, Jian and Zeng, Tieyong.*<br>
TPAMI 2022. [[Paper](https://ieeexplore.ieee.org/abstract/document/9785970)]
15. **Learning from imbalanced data.**<br>
*He, Haibo and Garcia, Edwardo A.*<br>
TKDE 2009. [[Paper](https://ieeexplore.ieee.org/abstract/document/5128907)]
16. **Deep long-tailed learning: A survey.**<br>
*Zhang, Yifan and Kang, Bingyi and Hooi, Bryan and Yan, Shuicheng and Feng, Jiashi.*<br>
TPAMI 2023. [[Paper](https://ieeexplore.ieee.org/abstract/document/10105457)]
17. **Image classification with deep learning in the presence of noisy labels: A survey.**<br>
*Algan, Görkem and Ulusoy, Ilkay.*<br>
Knowledge-Based Systems 2021. [[Paper](https://www.sciencedirect.com/science/article/pii/S0950705121000344?casa_token=TulsH2oHCicAAAAA:JLPtw-N9KqDyATVgTNxICU4l4yXVgWwk1d8NaPCwAULWB84o7OCSQSWrKPa6Z1Dc8YcHvwyjHvc)]
18. **Learning from noisy labels with deep neural networks: A survey.**<br>
*Song, Hwanjun and Kim, Minseok and Park, Dongmin and Shin, Yooju and Lee, Jae-Gil.*<br>
TNNLS 2022. [[Paper](https://ieeexplore.ieee.org/abstract/document/9729424)]
19. **A Survey of Learning on Small Data.**<br>
*Cao, Xiaofeng and Bu, Weixin and Huang, Shengjun and Tang, Yingpeng and Guo, Yaming and Chang, Yi and Tsang, Ivor W.*<br>
arXiv 2022. [[Paper](https://arxiv.org/abs/2207.14443)]
20. **Generalizing from a Few Examples: A Survey on Few-shot Learning.**<br>
*Wang, Yaqing and Yao, Quanming and Kwok, James T. and Ni, Lionel M.*<br>
ACM computing surveys 2021. [[Paper](https://dl.acm.org/doi/10.1145/3386252)]
22. **Learning under concept drift: A review.**<br>
*Lu, Jie and Liu, Anjin and Dong, Fan and Gu, Feng and Gama, Joao and Zhang, Guangquan.*<br>
TKDE 2018. [[Paper](https://ieeexplore.ieee.org/abstract/document/8496795)]
23. **Recent Advances in Concept Drift Adaptation Methods for Deep Learning.**<br>
*Yuan, Liheng and Li, Heng and Xia, Beihao and Gao, Cuiying and Liu, Mingyue and Yuan, Wei and You, Xinge.*<br>
IJCAI 2022. [[Paper](https://www.ijcai.org/proceedings/2022/0788.pdf)]
24. **Adaptive dendritic cell-deep learning approach for industrial prognosis under changing conditions.**<br>
*Diez-Olivan, Alberto and Ortego, Patxi and Del Ser, Javier and Landa-Torres, Itziar and Galar, Diego and Camacho, David and Sierra, Basilio.*<br>
IEEE Transactions on Industrial Informatics 2021. [[Paper](https://ieeexplore.ieee.org/abstract/document/9352529)]
25. **A survey on concept drift adaptation.**<br>
*Gama, João and Žliobaitė, Indrė and Bifet, Albert and Pechenizkiy, Mykola and Bouchachia, Abdelhamid.*<br>
ACM computing surveys 2014. [[Paper](https://dl.acm.org/doi/abs/10.1145/2523813)]
26. **An overview on concept drift learning.**<br>
*Iwashita, Adriana Sayuri and Papa, Joao Paulo.*<br>
IEEE access 2018. [[Paper](https://ieeexplore.ieee.org/abstract/document/8571222)]
27. **Opportunities and challenges in deep learning adversarial robustness: A survey.**<br>
*Silva, Samuel Henrique and Najafirad, Peyman.*<br>
arXiv 2020. [[Paper](https://arxiv.org/abs/2007.00753)]
28. **Robustness of deep learning models on graphs: A survey.**<br>
*Xu, Jiarong and Chen, Junru and You, Siqi and Xiao, Zhiqing and Yang, Yang and Lu, Jiangang.*<br>
AI Open 2021. [[Paper](https://www.sciencedirect.com/science/article/pii/S2666651021000139)]
29. **A survey of adversarial defenses and robustness in nlp.**<br>
*Goyal, Shreya and Doddapaneni, Sumanth and Khapra, Mitesh M and Ravindran, Balaraman.*<br>
ACM Computing Surveys 2023. [[Paper](https://dl.acm.org/doi/abs/10.1145/3593042)]
30. **A survey on bias and fairness in machine learning.**<br>
*Mehrabi, Ninareh and Morstatter, Fred and Saxena, Nripsuta and Lerman, Kristina and Galstyan, Aram.*<br>
ACM computing surveys 2021. [[Paper](https://dl.acm.org/doi/abs/10.1145/3457607)]
31. **FAIR: Fair adversarial instance re-weighting.**<br>
*Petrović, Andrija and Nikolić, Mladen and Radovanović, Sandro and Delibašić, Boris and Jovanović, Miloš.*<br>
Neurocomputing 2022. [[Paper](https://www.sciencedirect.com/science/article/pii/S0925231221019408?casa_token=zl0smR7i06AAAAAA:ybSefSP57QrNHVMLB9lb4rTQLCubIPA2Ggnh87bSC3Dv4faAC4f2zg5a38HQwA-6OyDUVpIK4C4)]
32. **Trustworthiness of autonomous systems.**<br>
*Devitt, S.*<br>
Foundations of trusted autonomy 2018. [[Paper](https://link.springer.com/chapter/10.1007/978-3-319-64816-3_9)]
33. **Trustworthy artificial intelligence: a review.**<br>
*Kaur, Davinder and Uslu, Suleyman and Rittichier, Kaley J and Durresi, Arjan.*<br>
ACM Computing Surveys 2022. [[Paper](https://dl.acm.org/doi/abs/10.1145/3491209)]
34. **Trustworthy Graph Learning: Reliability, Explainability, and Privacy Protection.**<br>
*Wu, Bingzhe and Bian, Yatao and Zhang, Hengtong and Li, Jintang and Yu, Junchi and Chen, Liang and Chen, Chaochao and Huang, Junzhou.*<br>
ACM KDD 2022. [[Paper](https://dl.acm.org/doi/abs/10.1145/3534678.3542597)]
35. **Combating Noisy Labels in Long-Tailed Image Classification.**<br>
*Fang, Chaowei and Cheng, Lechao and Qi, Huiyan and Zhang, Dingwen.*<br>
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.00273)]
36. **An Empirical Study of Accuracy, Fairness, Explainability, Distributional Robustness, and Adversarial Robustness.**<br>
*Singh, Moninder and Ghalachyan, Gevorg and Varshney, Kush R and Bryant, Reginald E.*<br>
arXiv 2021. [[Paper](https://arxiv.org/abs/2109.14653)]
37. **A Survey of Data Optimization for Problems in Computer Vision Datasets.**<br>
*Wan, Zhijing and Wang, Zhixiang and Chung, CheukTing and Wang, Zheng.*<br>
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.11717)]
38. **Towards Data-centric Graph Machine Learning: Review and Outlook.**<br>
*Zheng, Xin and Liu, Yixin and Bao, Zhifeng and Fang, Meng and Hu, Xia and Liew, Alan Wee-Chung and Pan, Shirui.*<br>
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.10979)]
# Overall of The proposed taxonomy
# Goals, scenarios, and data objects
## Optimization goals
39. **G-softmax: improving intraclass compactness and interclass separability of features.**<br>
*Luo, Yan and Wong, Yongkang and Kankanhalli, Mohan and Zhao, Qi.*<br>
TNNLS 2019. [[Paper](https://ieeexplore.ieee.org/ielaam/5962385/8984609/8712413-aam.pdf)]
40. **Label noise sgd provably prefers flat global minimizers.**<br>
*Damian, Alex and Ma, Tengyu and Lee, Jason D.*<br>
NeurIPS 2021. [[Paper](https://proceedings.neurips.cc/paper/2021/file/e6af401c28c1790eaef7d55c92ab6ab6-Paper.pdf)]
41. **Implicit semantic data augmentation for deep networks.**<br>
*Wang, Yulin and Pan, Xuran and Song, Shiji and Zhang, Hong and Huang, Gao and Wu, Cheng.*<br>
NeurIPS 2019. [[Paper](https://proceedings.neurips.cc/paper/2019/file/15f99f2165aa8c86c9dface16fefd281-Paper.pdf)]
42. **Meta balanced network for fair face recognition.**<br>
*Wang, Mei and Zhang, Yaobin and Deng, Weihong.*<br>
TPAMI 2021. [[Paper](https://ieeexplore.ieee.org/abstract/document/9512390)]
43. **Data Augmentation by Selecting Mixed Classes Considering Distance Between Classes.**<br>
*Fujii, Shungo and Ishii, Yasunori and Kozuka, Kazuki and Hirakawa, Tsubasa and Yamashita, Takayoshi and Fujiyoshi, Hironobu.*<br>
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.05122)]
44. **mixup: Beyond Empirical Risk Minimization.**<br>
*Zhang, Hongyi and Cisse, Moustapha and Dauphin, Yann N and Lopez-Paz, David.*<br>
ICLR 2018. [[Paper](https://openreview.net/pdf?id=r1Ddp1-Rb)]
45. **Online batch selection for faster training of neural networks.**<br>
*Loshchilov, Ilya and Hutter, Frank.*<br>
ICLR workshop track 2016. [[Paper](https://openreview.net/forum?id=r8lrkABJ7H8wknpYt5KB)]
46. **Fair Mixup: Fairness via Interpolation.**<br>
*Mroueh, Youssef and others.*<br>
ICLR 2021. [[Paper](https://openreview.net/pdf?id=DNl5s5BXeBn)]
47. **Fair classification with adversarial perturbations.**<br>
*Celis, L Elisa and Mehrotra, Anay and Vishnoi, Nisheeth.*<br>
NeurIPS 2021. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2021/file/44e207aecc63505eb828d442de03f2e9-Paper.pdf)]
48. **FORML: Learning to Reweight Data for Fairness.**<br>
*Yan, Bobby and Seto, Skyler and Apostoloff, Nicholas.*<br>
arXiv 2022. [[Paper](https://arxiv.org/abs/2202.01719)]
49. **Class-balanced loss based on effective number of samples.**<br>
*Cui, Yin and Jia, Menglin and Lin, Tsung-Yi and Song, Yang and Belongie, Serge.*<br>
CVPR 2019. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf)]
50. **Logit perturbation.**<br>
*Li, Mengyang and Su, Fengguang and Wu, Ou and Zhang, Ji.*<br>
AAAI 2022. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/20024)]
51. **Scale-aware automatic augmentations for object detection with dynamic training.**<br>
*Chen, Yukang and Zhang, Peizhen and Kong, Tao and Li, Yanwei and Zhang, Xiangyu and Qi, Lu and Sun, Jian and Jia, Jiaya.*<br>
TPAMI 2023. [[Paper](https://ieeexplore.ieee.org/abstract/document/9756374)]
52. **Obtaining well calibrated probabilities using bayesian binning.**<br>
*Naeini, Mahdi Pakdaman and Cooper, Gregory and Hauskrecht, Milos.*<br>
AAAI 2015. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/9602)]
53. **The devil is in the margin: Margin-based label smoothing for network calibration.**<br>
*Liu, Bingyuan and Ben Ayed, Ismail and Galdran, Adrian and Dolz, Jose.*<br>
CVPR 2022. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_The_Devil_Is_in_the_Margin_Margin-Based_Label_Smoothing_for_CVPR_2022_paper.pdf)]
54. **Calibrating deep neural networks using focal loss.**<br>
*Mukhoti, Jishnu and Kulharia, Viveka and Sanyal, Amartya and Golodetz, Stuart and Torr, Philip and Dokania, Puneet.*<br>
NeurIPS 2020. [[Paper](https://proceedings.neurips.cc/paper/2020/file/aeb7b30ef1d024a76f21a1d40e30c302-Paper.pdf)]
## Application scenarios
55. **Can Data Diversity Enhance Learning Generalization?**<br>
*Yu, Yu and Khadivi, Shahram and Xu, Jia.*<br>
COLING 2022. [[Paper](https://aclanthology.org/2022.coling-1.437.pdf)]
56. **Diversify your vision datasets with automatic diffusion-based augmentation.**<br>
*Dunlap, Lisa and Umino, Alyssa and Zhang, Han and Yang, Jiezhi and Gonzalez, Joseph E and Darrell, Trevor.*<br>
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16289)]
57. **Infusing definiteness into randomness: rethinking composition styles for deep image matting.**<br>
*Ye, Zixuan and Dai, Yutong and Hong, Chaoyi and Cao, Zhiguo and Lu, Hao.*<br>
AAAI 2023. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/25432)]
58. **Image data augmentation for deep learning: A survey.**<br>
*Yang, Suorong and Xiao, Weikang and Zhang, Mengcheng and Guo, Suhan and Zhao, Jian and Shen, Furao.*<br>
arXiv 2022. [[Paper](https://arxiv.org/abs/2204.08610)]
59. **BigTranslate: Augmenting Large Language Models with Multilingual Translation Capability over 100 Languages.**<br>
*Yang, Wen and Li, Chong and Zhang, Jiajun and Zong, Chengqing.*<br>
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18098)]
60. **Adversarial training for large neural language models.**<br>
*Liu, Xiaodong and Cheng, Hao and He, Pengcheng and Chen, Weizhu and Wang, Yu and Poon, Hoifung and Gao, Jianfeng.*<br>
arXiv 2020. [[Paper](https://arxiv.org/abs/2004.08994)]
## Data objects
61. **Understanding the difficulty of training deep feedforward neural networks.**<br>
*Glorot, Xavier and Bengio, Yoshua.*<br>
AISTATS 2010. [[Paper](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)]
62. **Delving deep into rectifiers: Surpassing human-level performance on imagenet classification.**<br>
*He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian.*<br>
ICCV 2015. [[Paper](https://openaccess.thecvf.com/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf)]
63. **Submodular Meta Data Compiling for Meta Optimization.**<br>
*Su, Fengguang and Zhu, Yu and Wu, Ou and Deng, Yingjun.*<br>
ECML/PKDD 2022. [[Paper](https://2022.ecmlpkdd.org/wp-content/uploads/2022/09/sub_474.pdf)]
# Optimization pipeline
## Data perception 
### Perception on different granularity levels
64. **O2u-net: A simple noisy label detection approach for deep neural networks.**<br>
*Huang, Jinchi and Qu, Lie and Jia, Rongfei and Zhao, Binqiang.*<br>
ICCV 2019. [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_O2U-Net_A_Simple_Noisy_Label_Detection_Approach_for_Deep_Neural_ICCV_2019_paper.pdf)] 
65. **Gradient harmonized single-stage detector.**<br>
*Li, Buyu and Liu, Yu and Wang, Xiaogang.*<br>
AAAI 2019. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/4877)]
66. **Delving deep into label smoothing.**<br>
*Zhang, Chang-Bin and Jiang, Peng-Tao and Hou, Qibin and Wei, Yunchao and Han, Qi and Li, Zhen and Cheng, Ming-Ming.*<br>
TIP 2021. [[Paper](https://ieeexplore.ieee.org/abstract/document/9464693)]
67. **Class-wise difficulty-balanced loss for solving class-imbalance.**<br>
*Sinha, Saptarshi and Ohashi, Hiroki and Nakamura, Katsuyuki.*<br>
ACCV 2020. [[Paper](https://openaccess.thecvf.com/content/ACCV2020/papers/Sinha_Class-Wise_Difficulty-Balanced_Loss_for_Solving_Class-Imbalance_ACCV_2020_paper.pdf)]
68. **Ccl: Class-wise curriculum learning for class imbalance problems.**<br>
*Escudero-Viñolo, Marcos and López-Cifuentes, Alejandro.*<br>
ICIP 2022. [[Paper](https://ieeexplore.ieee.org/abstract/document/9897273)]
69. **Hyper-sausage coverage function neuron model and learning algorithm for image classification.**<br>
*Ning, Xin and Tian, Weijuan and He, Feng and Bai, Xiao and Sun, Le and Li, Weijun.*<br>
Pattern Recognition 2023. [[Paper](https://www.sciencedirect.com/science/article/pii/S0031320322006951)]
70. **Measuring the effect of training data on deep learning predictions via randomized experiments.**<br>
*Lin, Jinkun and Zhang, Anqi and Lécuyer, Mathias and Li, Jinyang and Panda, Aurojit and Sen, Siddhartha.*<br>
ICML 2022. [[Paper](https://proceedings.mlr.press/v162/lin22h/lin22h.pdf)]
71. **Combining Adversaries with Anti-adversaries in Training.**<br>
*Zhou, Xiaoling and Yang, Nan and Wu, Ou.*<br>
AAAI 2023. [[Paper](https://arxiv.org/abs/2304.12550)]
### Perception on different types                        
72. **Implicit semantic data augmentation for deep networks.**<br>
*Wang, Yulin and Pan, Xuran and Song, Shiji and Zhang, Hong and Huang, Gao and Wu, Cheng.*<br>
NeurIPS 2019. [[Paper](https://proceedings.neurips.cc/paper/2019/file/15f99f2165aa8c86c9dface16fefd281-Paper.pdf)]
73. **Invariant feature learning for generalized long-tailed classification.**<br>
*Tang, Kaihua and Tao, Mingyuan and Qi, Jiaxin and Liu, Zhenguang and Zhang, Hanwang.*<br>
ECCV 2022. [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-20053-3_41)]
74. **DatasetEquity: Are All Samples Created Equal? In The Quest For Equity Within Datasets.**<br>
*Shrivastava, Shubham and Zhang, Xianling and Nagesh, Sushruth and Parchami, Armin.*<br>
ICCV 2023. [[Paper](https://openaccess.thecvf.com/content/ICCV2023W/OODCV/papers/Shrivastava_DatasetEquity_Are_All_Samples_Created_Equal_In_The_Quest_For_ICCVW_2023_paper.pdf)]
75. **Tackling the imbalance for gnns.**<br>
*Wang, Rui and Xiong, Weixuan and Hou, Qinghu and Wu, Ou.*<br>
IJCNN 2022. [[Paper](https://ieeexplore.ieee.org/abstract/document/9892713)]
76. **O2u-net: A simple noisy label detection approach for deep neural networks.**<br>
*Huang, Jinchi and Qu, Lie and Jia, Rongfei and Zhao, Binqiang.*<br>
ICCV 2019. [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_O2U-Net_A_Simple_Noisy_Label_Detection_Approach_for_Deep_Neural_ICCV_2019_paper.pdf)]
77. **Curriculum learning.**<br>
*Bengio, Yoshua and Louradour, Jérôme and Collobert, Ronan and Weston, Jason.*<br>
ICML 2009. [[Paper](https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/15972/Bengio%2C%202009%20Curriculum%20Learning.pdf?sequence=1&isAllowed=y)]
78. **Focal loss for dense object detection.**<br>
*Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Dollár, Piotr.*<br>
ICCV 2017. [[Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)]
79. **Deep learning on a data diet: Finding important examples early in training.**<br>
*Paul, Mansheej and Ganguli, Surya and Dziugaite, Gintare Karolina.*<br>
NeurIPS 2021. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2021/file/ac56f8fe9eea3e4a365f29f0f1957c55-Paper.pdf)]
80. **Exploring the Learning Difficulty of Data Theory and Measure.**<br>
*Zhu, Weiyao and Wu, Ou and Su, Fengguang and Deng, Yingjun.*<br>
arXiv 2022. [[Paper](https://arxiv.org/abs/2205.07427)]
81. **Beyond neural scaling laws: beating power law scaling via data pruning.**<br>
*Sorscher, Ben and Geirhos, Robert and Shekhar, Shashank and Ganguli, Surya and Morcos, Ari.*<br>
NeurIPS 2022. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/7b75da9b61eda40fa35453ee5d077df6-Paper-Conference.pdf)]
82. **A review of uncertainty quantification in deep learning: Techniques, applications and challenges.**<br>
*Abdar, Moloud and Pourpanah, Farhad and Hussain, Sadiq and Rezazadegan, Dana and Liu, Li and Ghavamzadeh, Mohammad and Fieguth, Paul and Cao, Xiaochun and Khosravi, Abbas and Acharya, U Rajendra and Makarenkov, Vladimir and Nahavandi, Saeid.*<br>
Information fusion 2021. [[Paper](https://www.sciencedirect.com/science/article/pii/S1566253521001081)]
83. **A tale of two long tails.**<br>
*D'souza, Daniel and Nussbaum, Zach and Agarwal, Chirag and Hooker, Sara.*<br>
arXiv 2021. [[Paper](https://arxiv.org/abs/2107.13098)]
84. **What uncertainties do we need in bayesian deep learning for computer vision?**<br>
*Kendall, Alex and Gal, Yarin.*<br>
NeurIPS 2017. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/2650d6089a6d640c5e85b2b88265dc2b-Paper.pdf)]
85. **Submodular optimization-based diverse paraphrasing and its effectiveness in data augmentation.**<br>
*Kumar, Ashutosh and Bhattamishra, Satwik and Bhandari, Manik and Talukdar, Partha.*<br>
NAACL 2019. [[Paper](https://aclanthology.org/N19-1363/)]
86. **Submodular Meta Data Compiling for Meta Optimization.**<br>
*Su, Fengguang and Zhu, Yu and Wu, Ou and Deng, Yingjun.*<br>
ECML/PKDD 2022. [[Paper](https://2022.ecmlpkdd.org/wp-content/uploads/2022/09/sub_474.pdf)]
87. **The vendi score: A diversity evaluation metric for machine learning.**<br>
*Friedman, Dan and Dieng, Adji Bousso.*<br>
arXiv 2023. [[Paper](https://arxiv.org/abs/2210.02410)]
88. **Improved techniques for training gans.**<br>
*Salimans, Tim and Goodfellow, Ian and Zaremba, Wojciech and Cheung, Vicki and Radford, Alec and Chen, Xi.*<br>
NeurIPS 2016. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2016/file/8a3363abe792db2d8761d6403605aeb7-Paper.pdf)]
89. **Rethinking Class Imbalance in Machine Learning.**<br>
*Wu, Ou.*<br>
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.03900)]  
90. **Invariant feature learning for generalized long-tailed classification.**<br>
*Tang, Kaihua and Tao, Mingyuan and Qi, Jiaxin and Liu, Zhenguang and Zhang, Hanwang.*<br>
ECCV 2022. [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-20053-3_41)]
91. **Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics.**<br>
*Swayamdipta, Swabha and Schwartz, Roy and Lourie, Nicholas and Wang, Yizhong and Hajishirzi, Hannaneh and Smith, Noah A and Choi, Yejin.*<br>
EMNLP 2020. [[Paper](https://aclanthology.org/2020.emnlp-main.746/)]
92. **Learning with neighbor consistency for noisy labels.**<br>
*Iscen, Ahmet and Valmadre, Jack and Arnab, Anurag and Schmid, Cordelia.*<br>
CVPR 2022. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Iscen_Learning_With_Neighbor_Consistency_for_Noisy_Labels_CVPR_2022_paper.pdf)]
93. **An Empirical Study of Example Forgetting during Deep Neural Network Learning.**<br>
*Toneva, Mariya and Sordoni, Alessandro and des Combes, Remi Tachet and Trischler, Adam and Bengio, Yoshua and Gordon, Geoffrey J.*<br>
ICLR 2019. [[Paper](https://openreview.net/forum?id=BJlxm30cKm&fbclid=IwAR3kUvKWW-NyzCi7dB_zL47J_KJSMtcqQp8eFQEd5R07VWj5dcCwHJsXRcc)]
94. **Attaining Class-Level Forgetting in Pretrained Model Using Few Samples.**<br>
*Singh, Pravendra and Mazumder, Pratik and Karim, Mohammed Asad.*<br>
ECCV 2022. [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-19778-9_25)]
95. **Characterizing datapoints via second-split forgetting.**<br>
*Maini, Pratyush and Garg, Saurabh and Lipton, Zachary and Kolter, J Zico.*<br>
NeurIPS 2022. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/c20447998d6c624b4b97d4466a3bfff5-Paper-Conference.pdf)]  
96. **A comprehensive survey of forgetting in deep learning beyond continual learning.**<br>
*Wang, Zhenyi and Yang, Enneng and Shen, Li and Huang, Heng.*<br>
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.09218)]  
97. **FINE samples for learning with noisy labels.**<br>
*Kim, Taehyeon and Ko, Jongwoo and Cho, Sangwook and Choi, Jinhwan and Yun, Se-Young.*<br>
NeurIPS 2021. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2021/file/ca91c5464e73d3066825362c3093a45f-Paper.pdf)]  
98. **A value for n-person games.**<br>
*L. S. Shapley.*<br>
Contributions to the Theory of Games 1953. [[Paper](https://apps.dtic.mil/sti/tr/pdf/AD0604084.pdf)]
99. **Data shapley: Equitable valuation of data for machine learning.**<br>
*Ghorbani, Amirata and Zou, James.*<br>
ICML 2019. [[Paper](https://proceedings.mlr.press/v97/ghorbani19c/ghorbani19c.pdf)]     
100. **Data valuation using reinforcement learning.**<br>
*Yoon, Jinsung and Arik, Sercan and Pfister, Tomas.*<br>
ICML 2020. [[Paper](https://proceedings.mlr.press/v119/yoon20a/yoon20a.pdf)] 
101. **Measuring the effect of training data on deep learning predictions via randomized experiments.**<br>
*Lin, Jinkun and Zhang, Anqi and Lécuyer, Mathias and Li, Jinyang and Panda, Aurojit and Sen, Siddhartha.*<br>
ICML 2022. [[Paper](https://proceedings.mlr.press/v162/lin22h/lin22h.pdf)]   
102. **Energy-Based Learning for Cooperative Games, with Applications to Valuation Problems in Machine Learning.**<br>
*Bian, Yatao and Rong, Yu and Xu, Tingyang and Wu, Jiaxiang and Krause, Andreas and Huang, Junzhou.*<br>
ICLR 2021. [[Paper](https://openreview.net/forum?id=xLfAgCroImw)]  
103. **OpenDataVal: a Unified Benchmark for Data Valuation.**<br>
*Jiang, Kevin Fu and Liang, Weixin and Zou, James and Kwon, Yongchan.*<br>
NeurIPS 2023. [[Paper](https://arxiv.org/abs/2306.10577)] 
104. **Locally adaptive label smoothing improves predictive churn.**<br>
*Bahri, Dara and Jiang, Heinrich.*<br>
ICML 2021. [[Paper](https://proceedings.mlr.press/v139/bahri21a/bahri21a.pdf)] 
105. **Data Profiling for Adversarial Training: On the Ruin of Problematic Data.**<br>
*Dong, Chengyu and Liu, Liyuan and Shang, Jingbo.*<br>
arXiv 2021. [[Paper](https://arxiv.org/abs/2102.07437v1)] 
106. **Training data influence analysis and estimation: A survey.**<br>
*Hammoudeh, Zayd and Lowd, Daniel.*<br>
arXiv 2023. [[Paper](https://arxiv.org/abs/2212.04612)]  
107. **Learning to purify noisy labels via meta soft label corrector.**<br>
*Wu, Yichen and Shu, Jun and Xie, Qi and Zhao, Qian and Meng, Deyu.*<br>
AAAI 2021. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/17244)] 
108. **Meta-weight-net: Learning an explicit mapping for sample weighting.**<br>
*Shu, Jun and Xie, Qi and Yi, Lixuan and Zhao, Qian and Zhou, Sanping and Xu, Zongben and Meng, Deyu.*<br>
NeurIPS 2019. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/e58cc5ca94270acaceed13bc82dfedf7-Paper.pdf)]
109. **Combining Adversaries with Anti-adversaries in Training.**<br>
*Zhou, Xiaoling and Yang, Nan and Wu, Ou.*<br>
AAAI 2023. [[Paper](https://arxiv.org/abs/2304.12550)]
### Static and dynamic perception    
110. **O2u-net: A simple noisy label detection approach for deep neural networks.**<br>
*Huang, Jinchi and Qu, Lie and Jia, Rongfei and Zhao, Binqiang.*<br>
ICCV 2019. [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_O2U-Net_A_Simple_Noisy_Label_Detection_Approach_for_Deep_Neural_ICCV_2019_paper.pdf)]
111. **Self-paced learning for latent variable models.**<br>
*Kumar, M and Packer, Benjamin and Koller, Daphne.*<br>
NeurIPS 2010. [[Paper](https://proceedings.neurips.cc/paper/2010/file/e57c6b956a6521b28495f2886ca0977a-Paper.pdf)]                    
## Analysis on perceived quantities
112. **An Empirical Study of Example Forgetting during Deep Neural Network Learning.**<br>
*Toneva, Mariya and Sordoni, Alessandro and des Combes, Remi Tachet and Trischler, Adam and Bengio, Yoshua and Gordon, Geoffrey J.*<br>
ICLR 2019. [[Paper](https://openreview.net/forum?id=BJlxm30cKm&fbclid=IwAR3kUvKWW-NyzCi7dB_zL47J_KJSMtcqQp8eFQEd5R07VWj5dcCwHJsXRcc)]     
113. **O2u-net: A simple noisy label detection approach for deep neural networks.**<br>
*Huang, Jinchi and Qu, Lie and Jia, Rongfei and Zhao, Binqiang.*<br>
ICCV 2019. [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_O2U-Net_A_Simple_Noisy_Label_Detection_Approach_for_Deep_Neural_ICCV_2019_paper.pdf)]
114. **Exploring the Learning Difficulty of Data Theory and Measure.**<br>
*Zhu, Weiyao and Wu, Ou and Su, Fengguang and Deng, Yingjun.*<br>
arXiv 2022. [[Paper](https://arxiv.org/abs/2205.07427)]
115. **Unsupervised label noise modeling and loss correction.**<br>
*Arazo, Eric and Ortego, Diego and Albert, Paul and O’Connor, Noel and McGuinness, Kevin.*<br>
ICML 2019. [[Paper](https://proceedings.mlr.press/v97/arazo19a/arazo19a.pdf)]  
116. **MILD: Modeling the Instance Learning Dynamics for Learning with Noisy Labels.**<br>
*Hu, Chuanyang and Yan, Shipeng and Gao, Zhitong and He, Xuming.*<br>
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.11560)]                   
## Optimizing
## Data optimization techniques
## Data resampling 
117. **Repair: Removing representation bias by dataset resampling.**<br>
*Li, Yi and Vasconcelos, Nuno.*<br>
CVPR 2020. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_REPAIR_Removing_Representation_Bias_by_Dataset_Resampling_CVPR_2019_paper.pdf)]
118. **The class imbalance problem.**<br>
*Megahed, Fadel M and Chen, Ying-Ju and Megahed, Aly and Ong, Yuya and Altman, Naomi and Krzywinski, Martin.*<br>
Nature Methods 2021. [[Paper](https://www.nature.com/articles/s41592-021-01302-4)]
119. **Reslt: Residual learning for long-tailed recognition.**<br>
*Cui, Jiequan and Liu, Shu and Tian, Zhuotao and Zhong, Zhisheng and Jia, Jiaya.*<br>
TPAMI 2022. [[Paper](https://ieeexplore.ieee.org/abstract/document/9774921)]
120. **Batchbald: Efficient and diverse batch acquisition for deep bayesian active learning.**<br>
*Kirsch, Andreas and Van Amersfoort, Joost and Gal, Yarin.*<br>
NeurIPS 2020. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/95323660ed2124450caaac2c46b5ed90-Paper.pdf)]
121. **Online batch selection for faster training of neural networks.**<br>
*Loshchilov, Ilya and Hutter, Frank.*<br>
ICLR workshop track 2016. [[Paper](https://openreview.net/forum?id=r8lrkABJ7H8wknpYt5KB)]
122. **Learning from imbalanced data.**<br>
*He, Haibo and Garcia, Edwardo A.*<br>
TKDE 2009. [[Paper](https://ieeexplore.ieee.org/abstract/document/5128907)]
123. **Improving predictive inference under covariate shift by weighting the log-likelihood function.**<br>
*Shimodaira, Hidetoshi.*<br>
Journal of statistical planning and inference 2000. [[Paper](https://www.sciencedirect.com/science/article/pii/S0378375800001154?casa_token=rvwJ8e4TPt0AAAAA:TJUlHDHpcCd0-9xSlzt13K4hnlQQcF6Ed4e9JXzCBgAzQY8PPKah46j3f3QDZSedHP16vTFMVbw)]
124. **What is the effect of importance weighting in deep learning?**<br>
*Byrd, Jonathon and Lipton, Zachary.*<br>
ICML 2019. [[Paper](https://proceedings.mlr.press/v97/byrd19a/byrd19a.pdf)]    
125. **Black-box importance sampling.**<br>
*Liu, Qiang and Lee, Jason.*<br>
AISTATS 2017. [[Paper](https://proceedings.mlr.press/v54/liu17b/liu17b.pdf)]  
126. **Not all samples are created equal: Deep learning with importance sampling.**<br>
*Katharopoulos, Angelos and Fleuret, François.*<br>
ICML 2018. [[Paper](https://proceedings.mlr.press/v80/katharopoulos18a/katharopoulos18a.pdf)]  
127. **Gradient harmonized single-stage detector.**<br>
*Li, Buyu and Liu, Yu and Wang, Xiaogang.*<br>
AAAI 2019. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/4877)]
128. **Training deep models faster with robust, approximate importance sampling.**<br>
*Johnson, Tyler B and Guestrin, Carlos.*<br>
NeurIPS 2018. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2018/file/967990de5b3eac7b87d49a13c6834978-Paper.pdf)]  
129. **Accelerating deep learning by focusing on the biggest losers.**<br>
*Jiang, Angela H and Wong, Daniel L-K and Zhou, Giulio and Andersen, David G and Dean, Jeffrey and Ganger, Gregory R and Joshi, Gauri and Kaminksy, Michael and Kozuch, Michael and Lipton, Zachary C and Pillai, Padmanabhan.*<br>
arXiv 2020. [[Paper](https://arxiv.org/abs/1910.00762)]  
130. **Towards Understanding Deep Learning from Noisy Labels with Small-Loss Criterion.**<br>
*Gui, Xian-Jin and Wang, Wei and Tian, Zhang-Hao.*<br>
IJCAI 2021. [[Paper](https://www.ijcai.org/proceedings/2021/0340.pdf)]  
131. **Adaptiveface: Adaptive margin and sampling for face recognition.**<br>
*Liu, Hao and Zhu, Xiangyu and Lei, Zhen and Li, Stan Z.*<br>
CVPR 2019. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_AdaptiveFace_Adaptive_Margin_and_Sampling_for_Face_Recognition_CVPR_2019_paper.pdf)]                      
132. **Understanding the role of importance weighting for deep learning.**<br>
*Xu, Da and Ye, Yuting and Ruan, Chuanwei.*<br>
arXiv 2021. [[Paper](https://arxiv.org/abs/2103.15209)]  
133. **How to measure uncertainty in uncertainty sampling for active learning.**<br>
*Nguyen, Vu-Linh and Shaker, Mohammad Hossein and Hüllermeier, Eyke.*<br>
Machine Learning 2022. [[Paper](https://link.springer.com/article/10.1007/s10994-021-06003-9)]  
134. **A survey on uncertainty estimation in deep learning classification systems from a Bayesian perspective.**<br>
*Mena, José and Pujol, Oriol and Vitrià, Jordi.*<br>
ACM Computing Surveys 2021. [[Paper](https://diposit.ub.edu/dspace/bitstream/2445/183476/1/714838.pdf)]  
135. **Uncertainty aware sampling framework of weak-label learning for histology image classification.**<br>
*Aljuhani, Asmaa and Casukhela, Ishya and Chan, Jany and Liebner, David and Machiraju, Raghu.*<br>
MICCAI 2022. [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-16434-7_36)]
136. **Optimal subsampling with influence functions.**<br>
*Ting, Daniel and Brochu, Eric.*<br>
NeurIPS 2018. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2018/file/57c0531e13f40b91b3b0f1a30b529a1d-Paper.pdf)]
137. **Background data resampling for outlier-aware classification.**<br>
*Li, Yi and Vasconcelos, Nuno.*<br>
CVPR 2020. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Background_Data_Resampling_for_Outlier-Aware_Classification_CVPR_2020_paper.pdf)]  
138. **Sentence-level resampling for named entity recognition.**<br>
*Wang, Xiaochen and Wang, Yue.*<br>
NAACL 2022. [[Paper](https://aclanthology.org/2022.naacl-main.156/)]  
139. **Undersampling near decision boundary for imbalance problems.**<br>
*Zhang, Jianjun and Wang, Ting and Ng, Wing WY and Zhang, Shuai and Nugent, Chris D.*<br>
ICMLC 2019. [[Paper](https://www.researchgate.net/profile/Jianjun-Zhang-4/publication/338453773_Undersampling_Near_Decision_Boundary_for_Imbalance_Problems/links/5e6982f992851c20f321f55b/Undersampling-Near-Decision-Boundary-for-Imbalance-Problems.pdf)]  
140. **Autosampling: Search for effective data sampling schedules.**<br>
*Sun, Ming and Dou, Haoxuan and Li, Baopu and Yan, Junjie and Ouyang, Wanli and Cui, Lei.*<br>
ICML 2021. [[Paper](https://proceedings.mlr.press/v139/sun21a/sun21a.pdf)]  
## Data augmentation
141. **Understanding data augmentation in neural machine translation: Two perspectives towards generalization.**<br>
*Li, Guanlin and Liu, Lemao and Huang, Guoping and Zhu, Conghui and Zhao, Tiejun.*<br>
EMNLP-IJCNLP 2019. [[Paper](https://aclanthology.org/D19-1570/)]
142. **Maximum-entropy adversarial data augmentation for improved generalization and robustness.**<br>
*Zhao, Long and Liu, Ting and Peng, Xi and Metaxas, Dimitris.*<br>
NeurIPS 2020. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/a5bfc9e07964f8dddeb95fc584cd965d-Paper.pdf)]
143. **Data augmentation can improve robustness.**<br>
*Rebuffi, Sylvestre-Alvise and Gowal, Sven and Calian, Dan Andrei and Stimberg, Florian and Wiles, Olivia and Mann, Timothy A.*<br>
NeurIPS 2021. [[Paper](https://proceedings.neurips.cc/paper/2021/file/fb4c48608ce8825b558ccf07169a3421-Paper.pdf)]
144. **Data augmentation alone can improve adversarial training.**<br>
*Li, Lin and Spratling, Michael W.*<br>
ICLR 2023. [[Paper](https://openreview.net/forum?id=y4uc4NtTWaq)]
145. **A survey on data augmentation for text classification.**<br>
*Bayer, Markus and Kaufhold, Marc-André and Reuter, Christian.*<br>
ACM Computing Surveys 2022. [[Paper](https://dl.acm.org/doi/abs/10.1145/3544558)]
146. **A survey on image data augmentation for deep learning.**<br>
*Shorten, Connor and Khoshgoftaar, Taghi M.*<br>
Journal of big data 2019. [[Paper](https://journalofbigdata.springeropen.com/counter/pdf/10.1186/s40537-019-0197-0.pdf)]
147. **Data augmentation for deep graph learning: A survey.**<br>
*Ding, Kaize and Xu, Zhe and Tong, Hanghang and Liu, Huan.*<br>
ACM SIGKDD Explorations Newsletter 2022. [[Paper](https://dl.acm.org/doi/abs/10.1145/3575637.3575646)]
148. **Time Series Data Augmentation for Deep Learning: A Survey.**<br>
*Wen, Qingsong and Sun, Liang and Yang, Fan and Song, Xiaomin and Gao, Jingkun and Wang, Xue and Xu, Huan.*<br>
IJCAI 2021. [[Paper](https://www.ijcai.org/proceedings/2021/0631.pdf)]
### Sample/feature augmentation
149. **A survey on image data augmentation for deep learning.**<br>
*Shorten, Connor and Khoshgoftaar, Taghi M.*<br>
Journal of big data 2019. [[Paper](https://journalofbigdata.springeropen.com/counter/pdf/10.1186/s40537-019-0197-0.pdf)]
150. **Data augmentation approaches in natural language processing: A survey.**<br>
*Li, Bohan and Hou, Yutai and Che, Wanxiang.*<br>
Ai Open 2022. [[Paper](https://www.sciencedirect.com/science/article/pii/S2666651022000080)]
151. **Dataset augmentation in feature space.**<br>
*DeVries, Terrance and Taylor, Graham W.*<br>
ICLR Workshop track 2017. [[Paper](https://openreview.net/pdf?id=HJ9rLLcxg)]
152. **A simple feature augmentation for domain generalization.**<br>
*Li, Pan and Li, Da and Li, Wei and Gong, Shaogang and Fu, Yanwei and Hospedales, Timothy M.*<br>
ICCV 2021. [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_A_Simple_Feature_Augmentation_for_Domain_Generalization_ICCV_2021_paper.pdf)]
153. **Augmentation invariant and instance spreading feature for softmax embedding.**<br>
*Ye, Mang and Shen, Jianbing and Zhang, Xu and Yuen, Pong C and Chang, Shih-Fu.*<br>
TPAMI 2020. [[Paper](https://ieeexplore.ieee.org/abstract/document/9154587)]
154. **Feature space augmentation for long-tailed data.**<br>
*Chu, Peng and Bian, Xiao and Liu, Shaopeng and Ling, Haibin.*<br>
ECCV 2020. [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-58526-6_41)]
155. **Towards Deep Learning Models Resistant to Adversarial Attacks.**<br>
*Madry, Aleksander and Makelov, Aleksandar and Schmidt, Ludwig and Tsipras, Dimitris and Vladu, Adrian.*<br>
ICLR 2018. [[Paper](https://openreview.net/forum?id=rJzIBfZAb)]
156. **Recent Advances in Adversarial Training for Adversarial Robustness.**<br>
*Bai, Tao and Luo, Jinqi and Zhao, Jun and Wen, Bihan and Wang, Qian.*<br>
IJCAI 2021. [[Paper](https://www.ijcai.org/proceedings/2021/0591.pdf)]
157. **Graddiv: Adversarial robustness of randomized neural networks via gradient diversity regularization.**<br>
*Lee, Sungyoon and Kim, Hoki and Lee, Jaewook.*<br>
TPAMI 2022. [[Paper](https://ieeexplore.ieee.org/abstract/document/9761760)]
158. **Self-supervised label augmentation via input transformations.**<br>
*Lee, Hankook and Hwang, Sung Ju and Shin, Jinwoo.*<br>
ICML 2020. [[Paper](https://proceedings.mlr.press/v119/lee20c/lee20c.pdf)]
159. **Transductive label augmentation for improved deep network learning.**<br>
*Elezi, Ismail and Torcinovich, Alessandro and Vascon, Sebastiano and Pelillo, Marcello.*<br>
ICPR 2018. [[Paper](https://ieeexplore.ieee.org/abstract/document/8545524/)]
160. **Adversarial and isotropic gradient augmentation for image retrieval with text feedback.**<br>
*Huang, Fuxiang and Zhang, Lei and Zhou, Yuhang and Gao, Xinbo.*<br>
IEEE Transactions on Multimedia 2022. [[Paper](https://ieeexplore.ieee.org/abstract/document/9953564)]
### Explicit/implicit augmentation
161. **SMOTE: synthetic minority over-sampling technique}.**<br>
*Chawla, Nitesh V and Bowyer, Kevin W and Hall, Lawrence O and Kegelmeyer, W Philip.*<br>
Journal of artificial intelligence research 2002. [[Paper](https://www.jair.org/index.php/jair/article/view/10302)]
162. **A cost-sensitive deep learning-based approach for network traffic classification.**<br>
*Telikani, Akbar and Gandomi, Amir H and Choo, Kim-Kwang Raymond and Shen, Jun.*<br>
IEEE Transactions on Network and Service Management 2021. [[Paper](https://opus.lib.uts.edu.au/bitstream/10453/151147/2/368a20cf-97f8-48ab-8094-46fd31da71a9.pdf)]
163. **DeepSMOTE: Fusing deep learning and SMOTE for imbalanced data.**<br>
*Dablain, Damien and Krawczyk, Bartosz and Chawla, Nitesh V.*<br>
TNNLS 2022. [[Paper](https://ieeexplore.ieee.org/abstract/document/9694621)]
164. **mixup: Beyond Empirical Risk Minimization.**<br>
*Zhang, Hongyi and Cisse, Moustapha and Dauphin, Yann N and Lopez-Paz, David.*<br>
ICLR 2018. [[Paper](https://openreview.net/forum?id=r1Ddp1-Rb&;noteId=r1Ddp1-Rb)]
165. **Manifold mixup: Better representations by interpolating hidden states.**<br>
*Verma, Vikas and Lamb, Alex and Beckham, Christopher and Najafi, Amir and Mitliagkas, Ioannis and Lopez-Paz, David and Bengio, Yoshua.*<br>
ICML 2019. [[Paper](https://proceedings.mlr.press/v97/verma19a/verma19a.pdf)]
166. **Generative adversarial nets.**<br>
*Goodfellow, Ian and Pouget-Abadie, Jean and Mirza, Mehdi and Xu, Bing and Warde-Farley, David and Ozair, Sherjil and Courville, Aaron and Bengio, Yoshua.*<br>
NeurIPS 2014. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf)]
167. **A review on generative adversarial networks: Algorithms, theory, and applications.**<br>
*Gui, Jie and Sun, Zhenan and Wen, Yonggang and Tao, Dacheng and Ye, Jieping.*<br>
TKDE 2021. [[Paper](https://ieeexplore.ieee.org/abstract/document/9625798)]
168. **BAGAN: Data Augmentation with Balancing GAN.**<br>
*Mariani, Giovanni and Scheidegger, Florian and Istrate, Roxana and Bekas, Costas and Malossi, Cristiano.*<br>
ICML 2018. [[Paper](https://research.ibm.com/publications/bagan-data-augmentation-with-balancing-gan)]
169. **Auggan: Cross domain adaptation with gan-based data augmentation.**<br>
*Huang, Sheng-Wei and Lin, Che-Tsung and Chen, Shu-Ping and Wu, Yen-Yi and Hsu, Po-Hao and Lai, Shang-Hong.*<br>
ECCV 2018. [[Paper](https://openaccess.thecvf.com/content_ECCV_2018/html/Sheng-Wei_Huang_AugGAN_Cross_Domain_ECCV_2018_paper.html)]
170. **TS-GAN: Time-series GAN for Sensor-based Health Data Augmentation.**<br>
*Yang, Zhenyu and Li, Yantao and Zhou, Gang.*<br>
ACM Transactions on Computing for Healthcare 2023. [[Paper](https://dl.acm.org/doi/abs/10.1145/3583593)]
171. **Diffusion models: A comprehensive survey of methods and applications.**<br>
*Yang, Ling and Zhang, Zhilong and Song, Yang and Hong, Shenda and Xu, Runsheng and Zhao, Yue and Zhang, Wentao and Cui, Bin and Yang, Ming-Hsuan.*<br>
ACM Computing Surveys 2022. [[Paper](https://dl.acm.org/doi/abs/10.1145/3626235)]
172. **Multimodal Data Augmentation for Image Captioning using Diffusion Models.**<br>
*Xiao, Changrong and Xu, Sean Xin and Zhang, Kunpeng.*<br>
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.01855)]
173. **Diversify your vision datasets with automatic diffusion-based augmentation.**<br>
*Dunlap, Lisa and Umino, Alyssa and Zhang, Han and Yang, Jiezhi and Gonzalez, Joseph E and Darrell, Trevor.*<br>
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16289)]
174. **An Extensive Exploration of Back-Translation in 60 Languages.**<br>
*McNamee, Paul and Duh, Kevin.*<br>
Findings of ACL 2023. [[Paper](https://aclanthology.org/2023.findings-acl.518/)]
175. **I2t2i: Learning text to image synthesis with textual data augmentation.**<br>
*Dong, Hao and Zhang, Jingqing and McIlwraith, Douglas and Guo, Yike.*<br>
ICIP 2017. [[Paper](https://ieeexplore.ieee.org/abstract/document/8296635)]
176. **Set-level Guidance Attack: Boosting Adversarial Transferability of Vision-Language Pre-training Models.**<br>
*Lu, Dong and Wang, Zhiqiang and Wang, Teng and Guan, Weili and Gao, Hongchang and Zheng, Feng.*<br>
ICCV 2023. [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Lu_Set-level_Guidance_Attack_Boosting_Adversarial_Transferability_of_Vision-Language_Pre-training_Models_ICCV_2023_paper.pdf)]
177. **TTIDA: Controllable Generative Data Augmentation via Text-to-Text and Text-to-Image Models.**<br>
*Yin, Yuwei and Kaddour, Jean and Zhang, Xiang and Nie, Yixin and Liu, Zhenguang and Kong, Lingpeng and Liu, Qi.*<br>
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.08821)]
178. **Combining Adversaries with Anti-adversaries in Training.**<br>
*Zhou, Xiaoling and Yang, Nan and Wu, Ou.*<br>
AAAI 2023. [[Paper](https://arxiv.org/abs/2304.12550)]
179. **Improving generalization via uncertainty driven perturbations.**<br>
*Pagliardini, Matteo and Manunza, Gilberto and Jaggi, Martin and Jordan, Michael I and Chavdarova, Tatjana.*<br>
arXiv 2022. [[Paper](https://arxiv.org/abs/2202.05737)]
180. **Randaugment: Practical automated data augmentation with a reduced search space.**<br>
*Cubuk, Ekin D and Zoph, Barret and Shlens, Jonathon and Le, Quoc V.*<br>
CVPR workshops 2020. [[Paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w40/Cubuk_Randaugment_Practical_Automated_Data_Augmentation_With_a_Reduced_Search_Space_CVPRW_2020_paper.pdf)]
181. **Metamixup: Learning adaptive interpolation policy of mixup with metalearning.**<br>
*Mai, Zhijun and Hu, Guosheng and Chen, Dexiong and Shen, Fumin and Shen, Heng Tao.*<br>
TNNLS 2021. [[Paper](https://ieeexplore.ieee.org/abstract/document/9366422)]
182. **Automatic data augmentation via deep reinforcement learning for effective kidney tumor segmentation.**<br>
*Qin, Tiexin and Wang, Ziyuan and He, Kelei and Shi, Yinghuan and Gao, Yang and Shen, Dinggang.*<br>
ICASSP 2020. [[Paper](https://ieeexplore.ieee.org/abstract/document/9053403)]
183. **Augmentation strategies for learning with noisy labels.**<br>
*Nishi, Kento and Ding, Yi and Rich, Alex and Hollerer, Tobias.*<br>
CVPR 2021. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Nishi_Augmentation_Strategies_for_Learning_With_Noisy_Labels_CVPR_2021_paper.pdf)]
184. **Differentiable automatic data augmentation.**<br>
*Li, Yonggang and Hu, Guosheng and Wang, Yongtao and Hospedales, Timothy and Robertson, Neil M and Yang, Yongxin.*<br>
ECCV 2020. [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-58542-6_35)]
185. **Implicit semantic data augmentation for deep networks.**<br>
*Wang, Yulin and Pan, Xuran and Song, Shiji and Zhang, Hong and Huang, Gao and Wu, Cheng.*<br>
NeurIPS 2019. [[Paper](https://proceedings.neurips.cc/paper/2019/file/15f99f2165aa8c86c9dface16fefd281-Paper.pdf)]
186. **Imagine by reasoning: A reasoning-based implicit semantic data augmentation for long-tailed classification.**<br>
*Chen, Xiaohua and Zhou, Yucan and Wu, Dayan and Zhang, Wanqian and Zhou, Yu and Li, Bo and Wang, Weiping.*<br>
AAAI 2022. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/19912)]
187. **Implicit Counterfactual Data Augmentation for Deep Neural Networks.**<br>
*Zhou, Xiaoling and Wu, Ou.*<br>
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.13431)]
188. **On feature normalization and data augmentation.**<br>
*Li, Boyi and Wu, Felix and Lim, Ser-Nam and Belongie, Serge and Weinberger, Kilian Q.*<br>
CVPR 2021. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_On_Feature_Normalization_and_Data_Augmentation_CVPR_2021_paper.pdf)]
189. **Implicit rugosity regularization via data augmentation.**<br>
*LeJeune, Daniel and Balestriero, Randall and Javadi, Hamid and Baraniuk, Richard G.*<br>
arXiv 2019. [[Paper](https://arxiv.org/abs/1905.11639)]
190. **Mixup as locally linear out-of-manifold regularization.**<br>
*Guo, Hongyu and Mao, Yongyi and Zhang, Richong.*<br>
AAAI 2019. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/4256)]
191. **Avoiding overfitting: A survey on regularization methods for convolutional neural networks.**<br>
*Santos, Claudio Filipi Gonçalves Dos and Papa, João Paulo.*<br>
ACM Computing Surveys 2022. [[Paper](https://dl.acm.org/doi/full/10.1145/3510413)]
192. **The good, the bad and the ugly sides of data augmentation: An implicit spectral regularization perspective.**<br>
*Lin, Chi-Heng and Kaushik, Chiraag and Dyer, Eva L and Muthukumar, Vidya.*<br>
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.05021)]
193. **A group-theoretic framework for data augmentation.**<br>
*Chen, Shuxiao and Dobriban, Edgar and Lee, Jane H.*<br>
The Journal of Machine Learning Research 2020. [[Paper](https://dl.acm.org/doi/abs/10.5555/3455716.3455961)]
## Data perturbation
194. **Compensation learning.**<br>
*Yao, Rujing and Wu, Ou.*<br>
arXiv 2021. [[Paper](https://arxiv.org/abs/2107.11921)]
### Perturbation target
195. **Learn2perturb: an end-to-end feature perturbation learning to improve adversarial robustness.**<br>
*Jeddi, Ahmadreza and Shafiee, Mohammad Javad and Karg, Michelle and Scharfenberger, Christian and Wong, Alexander.*<br>
CVPR 2020. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jeddi_Learn2Perturb_An_End-to-End_Feature_Perturbation_Learning_to_Improve_Adversarial_Robustness_CVPR_2020_paper.pdf)]
196. **Encoding robustness to image style via adversarial feature perturbations.**<br>
*Shu, Manli and Wu, Zuxuan and Goldblum, Micah and Goldstein, Tom.*<br>
NeurIPS 2021. [[Paper](https://proceedings.neurips.cc/paper/2021/file/ec20019911a77ad39d023710be68aaa1-Paper.pdf)]
197. **Logit perturbation.**<br>
*Li, Mengyang and Su, Fengguang and Wu, Ou and Zhang, Ji.*<br>
AAAI 2022. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/20024)]
198. **Long-tail learning via logit adjustment.**<br>
*Menon, Aditya Krishna and Jayasumana, Sadeep and Rawat, Ankit Singh and Jain, Himanshu and Veit, Andreas and Kumar, Sanjiv.*<br>
ICLR 2021. [[Paper](https://openreview.net/pdf?id=37nvvqkCo5)]
199. **Learning imbalanced datasets with label-distribution-aware margin loss.**<br>
*Cao, Kaidi and Wei, Colin and Gaidon, Adrien and Arechiga, Nikos and Ma, Tengyu.*<br>
NeurIPS 2019. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/621461af90cadfdaf0e8d4cc25129f91-Paper.pdf)]
200. **Implicit semantic data augmentation for deep networks.**<br>
*Wang, Yulin and Pan, Xuran and Song, Shiji and Zhang, Hong and Huang, Gao and Wu, Cheng.*<br>
NeurIPS 2019. [[Paper](https://proceedings.neurips.cc/paper/2019/file/15f99f2165aa8c86c9dface16fefd281-Paper.pdf)]
201. **Class-Level Logit Perturbation.**<br>
*Li, Mengyang and Su, Fengguang and Wu, Ou and Zhang, Ji.*<br>
TNNLS 2023. [[Paper](https://ieeexplore.ieee.org/abstract/document/10130785/)]
202. **Rethinking the inception architecture for computer vision.**<br>
*Szegedy, Christian and Vanhoucke, Vincent and Ioffe, Sergey and Shlens, Jon and Wojna, Zbigniew.*<br>
CVPR 2016. [[Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)]
203. **Delving deep into label smoothing.**<br>
*Zhang, Chang-Bin and Jiang, Peng-Tao and Hou, Qibin and Wei, Yunchao and Han, Qi and Li, Zhen and Cheng, Ming-Ming.*<br>
TIP 2021. [[Paper](https://ieeexplore.ieee.org/abstract/document/9464693)]
204. **Adversarial robustness via label-smoothing.**<br>
*Goibert, Morgane and Dohmatob, Elvis.*<br>
arXiv 2019. [[Paper](https://arxiv.org/abs/1906.11567)]
205. **From label smoothing to label relaxation.**<br>
*Lienen, Julian and Hüllermeier, Eyke.*<br>
AAAI 2021. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/17041)]
206. **Anticorrelated noise injection for improved generalization.**<br>
*Orvieto, Antonio and Kersting, Hans and Proske, Frank and Bach, Francis and Lucchi, Aurelien.*<br>
ICML 2022. [[Paper](https://proceedings.mlr.press/v162/orvieto22a/orvieto22a.pdf)]
207. **Adversarial weight perturbation helps robust generalization.**<br>
*Wu, Dongxian and Xia, Shu-Tao and Wang, Yisen.*<br>
NeurIPS 2020. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/1ef91c212e30e14bf125e9374262401f-Paper.pdf)]
208. **Reinforcement learning with perturbed rewards.**<br>
*Wang, Jingkang and Liu, Yang and Li, Bo.*<br>
AAAI 2020. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/6086)]
### Perturbation direction
209. **Implicit semantic data augmentation for deep networks.**<br>
*Wang, Yulin and Pan, Xuran and Song, Shiji and Zhang, Hong and Huang, Gao and Wu, Cheng.*<br>
NeurIPS 2019. [[Paper](https://proceedings.neurips.cc/paper/2019/file/15f99f2165aa8c86c9dface16fefd281-Paper.pdf)]
210. **Combining Adversaries with Anti-adversaries in Training.**<br>
*Zhou, Xiaoling and Yang, Nan and Wu, Ou.*<br>
AAAI 2023. [[Paper](https://arxiv.org/abs/2304.12550)]
211. **Training deep neural networks on noisy labels with bootstrapping.**<br>
*Reed, Scott and Lee, Honglak and Anguelov, Dragomir and Szegedy, Christian and Erhan, Dumitru and Rabinovich, Andrew.*<br>
ICLR workshop 2015. [[Paper](https://imgtec.eetrend.com/sites/imgtec.eetrend.com/files/201709/blog/10381-29627-deep.pdf)]
212. **Logit perturbation.**<br>
*Li, Mengyang and Su, Fengguang and Wu, Ou and Zhang, Ji.*<br>
AAAI 2022. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/20024)]
### Perturbation granularity
213. **Universal adversarial training with class-wise perturbations.**<br>
*Benz, Philipp and Zhang, Chaoning and Karjauv, Adil and Kweon, In So.*<br>
ICME 2021. [[Paper](https://ieeexplore.ieee.org/abstract/document/9428419/)]
214. **Balancing Logit Variation for Long-tailed Semantic Segmentation.**<br>
*Wang, Yuchao and Fei, Jingjing and Wang, Haochen and Li, Wei and Bao, Tianpeng and Wu, Liwei and Zhao, Rui and Shen, Yujun.*<br>
CVPR 2023. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Balancing_Logit_Variation_for_Long-Tailed_Semantic_Segmentation_CVPR_2023_paper.pdf)]
215. **Universal adversarial training.**<br>
*Shafahi, Ali and Najibi, Mahyar and Xu, Zheng and Dickerson, John and Davis, Larry S and Goldstein, Tom.*<br>
AAAI 2020. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/6017)]
216. **Universal adversarial perturbations.**<br>
*Moosavi-Dezfooli, Seyed-Mohsen and Fawzi, Alhussein and Fawzi, Omar and Frossard, Pascal.*<br>
CVPR 2017. [[Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Moosavi-Dezfooli_Universal_Adversarial_Perturbations_CVPR_2017_paper.pdf)]
217. **Distribution-balanced loss for multi-label classification in long-tailed datasets.**<br>
*Wu, Tong and Huang, Qingqiu and Liu, Ziwei and Wang, Yu and Lin, Dahua.*<br>
ECCV 2020. [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-58548-8_10)]
### Assignment manner
218. **Transferable adversarial perturbations.**<br>
*Zhou, Wen and Hou, Xin and Chen, Yongjun and Tang, Mengyun and Huang, Xiangqi and Gan, Xiang and Yang, Yong.*<br>
ECCV 2018. [[Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Bruce_Hou_Transferable_Adversarial_Perturbations_ECCV_2018_paper.pdf)]
219. **Sparse adversarial perturbations for videos.**<br>
*Wei, Xingxing and Zhu, Jun and Yuan, Sha and Su, Hang.*<br>
AAAI 2019. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/4927)]
220. **Investigating annotation noise for named entity recognition.**<br>
*Zhu, Yu and Ye, Yingchun and Li, Mengyang and Zhang, Ji and Wu, Ou.*<br>
Neural Computing and Applications 2023. [[Paper](https://link.springer.com/article/10.1007/s00521-022-07733-0)]
221. **A simple framework for contrastive learning of visual representations.**<br>
*Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey.*<br>
ICML 2020. [[Paper](https://proceedings.mlr.press/v119/chen20j.html)]
222. **A self-supervised approach for adversarial robustness.**<br>
*Naseer, Muzammal and Khan, Salman and Hayat, Munawar and Khan, Fahad Shahbaz and Porikli, Fatih.*<br>
CVPR 2020. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Naseer_A_Self-supervised_Approach_for_Adversarial_Robustness_CVPR_2020_paper.pdf)]
223. **GANSER: A self-supervised data augmentation framework for EEG-based emotion recognition.**<br>
*Zhang, Zhi and Zhong, Sheng-hua and Liu, Yan.*<br>
IEEE Transactions on Affective Computing 2022. [[Paper](https://ieeexplore.ieee.org/abstract/document/9763358/)]
224. **Metasaug: Meta semantic augmentation for long-tailed visual recognition.**<br>
*Li, Shuang and Gong, Kaixiong and Liu, Chi Harold and Wang, Yulin and Qiao, Feng and Cheng, Xinjing.*<br>
CVPR 2021. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_MetaSAug_Meta_Semantic_Augmentation_for_Long-Tailed_Visual_Recognition_CVPR_2021_paper.pdf)]
225. **Uncertainty-guided model generalization to unseen domains.**<br>
*Qiao, Fengchun and Peng, Xi.*<br>
CVPR 2021. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Qiao_Uncertainty-Guided_Model_Generalization_to_Unseen_Domains_CVPR_2021_paper.pdf)]
226. **Autoaugment: Learning augmentation policies from data.**<br>
*Cubuk, Ekin D and Zoph, Barret and Mane, Dandelion and Vasudevan, Vijay and Le, Quoc V.*<br>
CVPR 2019. [[Paper](https://research.google/pubs/pub47890/)]
227. **Automatically Learning Data Augmentation Policies for Dialogue Tasks.**<br>
*Niu, Tong and Bansal, Mohit.*<br>
EMNLP 2019. [[Paper](https://aclanthology.org/D19-1132/)]
228. **Deep reinforcement adversarial learning against botnet evasion attacks.**<br>
*Apruzzese, Giovanni and Andreolini, Mauro and Marchetti, Mirco and Venturi, Andrea and Colajanni, Michele.*<br>
IEEE Transactions on Network and Service Management 2020. [[Paper](https://ieeexplore.ieee.org/abstract/document/9226405)]
229. **Adversarial reinforced instruction attacker for robust vision-language navigation.**<br>
*Lin, Bingqian and Zhu, Yi and Long, Yanxin and Liang, Xiaodan and Ye, Qixiang and Lin, Liang.*<br>
TPAMI 2021. [[Paper](https://ieeexplore.ieee.org/abstract/document/9488322)]
## Data weighting
### Weighting granularity
230. **Denoising implicit feedback for recommendation.**<br>
*Wang, Wenjie and Feng, Fuli and He, Xiangnan and Nie, Liqiang and Chua, Tat-Seng.*<br>
WSDM 2021. [[Paper](https://dl.acm.org/doi/abs/10.1145/3437963.3441800)]
231. **Superloss: A generic loss for robust curriculum learning.**<br>
*Castells, Thibault and Weinzaepfel, Philippe and Revaud, Jerome.*<br>
NeurIPS 2020. [[Paper](https://proceedings.neurips.cc/paper/2020/file/2cfa8f9e50e0f510ede9d12338a5f564-Paper.pdf)]
232. **Class-balanced loss based on effective number of samples.**<br>
*Cui, Yin and Jia, Menglin and Lin, Tsung-Yi and Song, Yang and Belongie, Serge.*<br>
CVPR 2019. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf)]
233. **Distribution alignment: A unified framework for long-tail visual recognition.**<br>
*Zhang, Songyang and Li, Zeming and Yan, Shipeng and He, Xuming and Sun, Jian.*<br>
CVPR 2021. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Distribution_Alignment_A_Unified_Framework_for_Long-Tail_Visual_Recognition_CVPR_2021_paper.pdf)]
234. **Dynamically weighted balanced loss: class imbalanced learning and confidence calibration of deep neural networks.**<br>
*Fernando, K Ruwani M and Tsokos, Chris P.*<br>
TNNLS 2021. [[Paper](https://ieeexplore.ieee.org/abstract/document/9324926)]
235. **Just train twice: Improving group robustness without training group information.**<br>
*Liu, Evan Z and Haghgoo, Behzad and Chen, Annie S and Raghunathan, Aditi and Koh, Pang Wei and Sagawa, Shiori and Liang, Percy and Finn, Chelsea.*<br>
ICML 2021. [[Paper](https://proceedings.mlr.press/v139/liu21f/liu21f.pdf)]
236. **Geometry-aware Instance-reweighted Adversarial Training.**<br>
*Zhang, Jingfeng and Zhu, Jianing and Niu, Gang and Han, Bo and Sugiyama, Masashi and Kankanhalli, Mohan.*<br>
ICLR 2021. [[Paper](https://openreview.net/forum?id=iAX0l6Cz8ub)]
237. **Credit card fraud detection: a realistic modeling and a novel learning strategy.**<br>
*Dal Pozzolo, Andrea and Boracchi, Giacomo and Caelen, Olivier and Alippi, Cesare and Bontempi, Gianluca.*<br>
TNNLS 2017. [[Paper](https://ieeexplore.ieee.org/abstract/document/8038008)]
238. **Cost-sensitive portfolio selection via deep reinforcement learning.**<br>
*Zhang, Yifan and Zhao, Peilin and Wu, Qingyao and Li, Bin and Huang, Junzhou and Tan, Mingkui.*<br>
TKDE 2020. [[Paper](https://ieeexplore.ieee.org/abstract/document/9031418)]
239. **Integrating TANBN with cost sensitive classification algorithm for imbalanced data in medical diagnosis.**<br>
*Gan, Dan and Shen, Jiang and An, Bang and Xu, Man and Liu, Na.*<br>
Computers & Industrial Engineering 2020. [[Paper](https://www.sciencedirect.com/science/article/pii/S0360835219307351?casa_token=97voI68djkMAAAAA:cRy98l9KsYxqelE8TlpklR7e7RcZD2dz9VvkF0Eg6FvwXAwvrCjJKfTbyzREOuY-TtDae5Hroiw)]
240. **FORML: Learning to Reweight Data for Fairness.**<br>
*Yan, Bobby and Seto, Skyler and Apostoloff, Nicholas.*<br>
arXiv 2022. [[Paper](https://arxiv.org/abs/2202.01719)]
241. **Fairness in graph mining: A survey.**<br>
*Dong, Yushun and Ma, Jing and Wang, Song and Chen, Chen and Li, Jundong.*<br>
TKDE 2023. [[Paper](https://ieeexplore.ieee.org/abstract/document/10097603)]
### Dependent factor
242. **Curriculum learning.**<br>
*Bengio, Yoshua and Louradour, Jérôme and Collobert, Ronan and Weston, Jason.*<br>
ICML 2009. [[Paper](https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/15972/Bengio%2C%202009%20Curriculum%20Learning.pdf?sequence=1&isAllowed=y)]
243. **Self-paced learning for latent variable models.**<br>
*Kumar, M and Packer, Benjamin and Koller, Daphne.*<br>
NeurIPS 2010. [[Paper](https://proceedings.neurips.cc/paper/2010/file/e57c6b956a6521b28495f2886ca0977a-Paper.pdf)]
244. **Easy samples first: Self-paced reranking for zero-example multimedia search.**<br>
*Jiang, Lu and Meng, Deyu and Mitamura, Teruko and Hauptmann, Alexander G.*<br>
ACM MM 2014. [[Paper](https://dl.acm.org/doi/abs/10.1145/2647868.2654918)]
245. **Self-paced learning with diversity.**<br>
*Jiang, Lu and Meng, Deyu and Yu, Shoou-I and Lan, Zhenzhong and Shan, Shiguang and Hauptmann, Alexander.*<br>
NeurIPS 2014. [[Paper](https://proceedings.neurips.cc/paper/2014/file/c60d060b946d6dd6145dcbad5c4ccf6f-Paper.pdf)]
246. **A self-paced multiple-instance learning framework for co-saliency detection.**<br>
*Zhang, Dingwen and Meng, Deyu and Li, Chao and Jiang, Lu and Zhao, Qian and Han, Junwei.*<br>
ICCV 2015. [[Paper](https://openaccess.thecvf.com/content_iccv_2015/papers/Zhang_A_Self-Paced_Multiple-Instance_ICCV_2015_paper.pdf)]
247. **Curriculum learning: A survey.**<br>
*Soviany, Petru and Ionescu, Radu Tudor and Rota, Paolo and Sebe, Nicu.*<br>
IJCV 2022. [[Paper](https://link.springer.com/article/10.1007/s11263-022-01611-x)]
248. **Focal loss for dense object detection.**<br>
*Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Dollár, Piotr.*<br>
ICCV 2017. [[Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)]
249. **Geometry-aware Instance-reweighted Adversarial Training.**<br>
*Zhang, Jingfeng and Zhu, Jianing and Niu, Gang and Han, Bo and Sugiyama, Masashi and Kankanhalli, Mohan.*<br>
ICLR 2021. [[Paper](https://openreview.net/forum?id=iAX0l6Cz8ub)]
250. **LOW: Training deep neural networks by learning optimal sample weights.**<br>
*Santiago, Carlos and Barata, Catarina and Sasdelli, Michele and Carneiro, Gustavo and Nascimento, Jacinto C.*<br>
Pattern Recognition 2021. [[Paper](https://www.sciencedirect.com/science/article/pii/S0031320320303885?casa_token=4OHT8wlvtroAAAAA:PRX8tFrNiPLvbPzQ7Fgsu9k-gUmqdNCePi0JdJJQzHzQarTjTeeo3MsnAsrc4lDwRSlqdKDuZLQ)]
251. **Curriculum Learning with Diversity for Supervised Computer Vision Tasks.**<br>
*Soviany, Petru.*<br>
ICML Workshop 2020. [[Paper](https://openreview.net/forum?id=WH27bUkkzj)]
252. **Which Samples Should Be Learned First: Easy or Hard?**<br>
*Which Samples Should Be Learned First: Easy or Hard?*<br>
TNNLS 2023. [[Paper](https://ieeexplore.ieee.org/abstract/document/10155763)]
253. **Metacleaner: Learning to hallucinate clean representations for noisy-labeled visual recognition.**<br>
*Zhang, Weihe and Wang, Yali and Qiao, Yu.*<br>
CVPR 2019. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_MetaCleaner_Learning_to_Hallucinate_Clean_Representations_for_Noisy-Labeled_Visual_Recognition_CVPR_2019_paper.pdf)]
254. **Confident learning: Estimating uncertainty in dataset labels.**<br>
*Northcutt, Curtis and Jiang, Lu and Chuang, Isaac.*<br>
Journal of Artificial Intelligence Research 2021. [[Paper](https://www.jair.org/index.php/jair/article/view/12125)]
### Assignment manners
255. **Class-balanced loss based on effective number of samples.**<br>
*Cui, Yin and Jia, Menglin and Lin, Tsung-Yi and Song, Yang and Belongie, Serge.*<br>
CVPR 2019. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf)]
256. **Focal loss for dense object detection.**<br>
*Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Dollár, Piotr.*<br>
ICCV 2017. [[Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)]
257. **Umix: Improving importance weighting for subpopulation shift via uncertainty-aware mixup.**<br>
*Han, Zongbo and Liang, Zhipeng and Yang, Fan and Liu, Liu and Li, Lanqing and Bian, Yatao and Zhao, Peilin and Wu, Bingzhe and Zhang, Changqing and Yao, Jianhua.*<br>
NeurIPS 2022. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/f593c9c251d4d7cf14d4ab9861dfb7eb-Paper-Conference.pdf)]
258. **Classification with noisy labels by importance reweighting.**<br>
*Liu, Tongliang and Tao, Dacheng.*<br>
TPAMI 2015. [[Paper](https://ieeexplore.ieee.org/abstract/document/7159100)]
259. **Self-paced learning for latent variable models.**<br>
*Kumar, M and Packer, Benjamin and Koller, Daphne.*<br>
NeurIPS 2010. [[Paper](https://proceedings.neurips.cc/paper/2010/file/e57c6b956a6521b28495f2886ca0977a-Paper.pdf)]  
260. **Self-paced learning: An implicit regularization perspective.**<br>
*Fan, Yanbo and He, Ran and Liang, Jian and Hu, Baogang.*<br>
AAAI 2017. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/10809)]
261. **Adversarial reweighting for partial domain adaptation.**<br>
*Gu, Xiang and Yu, Xi and Sun, Jian and Xu, Zongben.*<br>
NeurIPS 2021. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2021/file/7ce3284b743aefde80ffd9aec500e085-Paper.pdf)]
262. **Reweighting Augmented Samples by Minimizing the Maximal Expected Loss.**<br>
*Yi, Mingyang and Hou, Lu and Shang, Lifeng and Jiang, Xin and Liu, Qun and Ma, Zhi-Ming.*<br>
ICLR 2021. [[Paper](https://openreview.net/forum?id=9G5MIc-goqB)]
263. **Learning to reweight examples for robust deep learning.**<br>
*Ren, Mengye and Zeng, Wenyuan and Yang, Bin and Urtasun, Raquel.*<br>
ICML 2018. [[Paper](https://proceedings.mlr.press/v80/ren18a/ren18a.pdf)]
264. **Meta-weight-net: Learning an explicit mapping for sample weighting.**<br>
*Shu, Jun and Xie, Qi and Yi, Lixuan and Zhao, Qian and Zhou, Sanping and Xu, Zongben and Meng, Deyu.*<br>
NeurIPS 2019. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/e58cc5ca94270acaceed13bc82dfedf7-Paper.pdf)]
265. **A probabilistic formulation for meta-weight-net.**<br>
*Zhao, Qian and Shu, Jun and Yuan, Xiang and Liu, Ziming and Meng, Deyu.*<br>
TNNLS 2023. [[Paper](https://ieeexplore.ieee.org/abstract/document/9525050)]
266. **Unsupervised Domain Adaptation for Text Classification via Meta Self-Paced Learning.**<br>
*Trung, Nghia Ngo and Van, Linh Ngo and Nguyen, Thien Huu.*<br>
COLING 2022. [[Paper](https://aclanthology.org/2022.coling-1.420/)]
267. **Meta self-paced learning for cross-modal matching.**<br>
*Wei, Jiwei and Xu, Xing and Wang, Zheng and Wang, Guoqing.*<br>
ACM MM 2021. [[Paper](https://dl.acm.org/doi/abs/10.1145/3474085.3475451)]
268. **Meta-reweighted regularization for unsupervised domain adaptation.**<br>
*Li, Shuang and Ma, Wenxuan and Zhang, Jinming and Liu, Chi Harold and Liang, Jian and Wang, Guoren.*<br>
TKDE 2023. [[Paper](https://ieeexplore.ieee.org/abstract/document/9546671)]
269. **Metaaugment: Sample-aware data augmentation policy learning.**<br>
*Zhou, Fengwei and Li, Jiawei and Xie, Chuanlong and Chen, Fei and Hong, Lanqing and Sun, Rui and Li, Zhenguo.*<br>
AAAI 2021. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/17324)]
270. **Automated Data Denoising for Recommendation.**<br>
*Ge, Yingqiang and Rahmani, Mostafa and Irissappane, Athirai and Sepulveda, Jose and Wang, Fei and Caverlee, James and Zhang, Yongfeng.*<br>
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.07070)]
271. **Rethinking importance weighting for deep learning under distribution shift.**<br>
*Fang, Tongtong and Lu, Nan and Niu, Gang and Sugiyama, Masashi.*<br>
NeurIPS 2020. [[Paper](https://proceedings.neurips.cc/paper/2020/file/8b9e7ab295e87570551db122a04c6f7c-Paper.pdf)]
## Data pruning
### Dataset distillation
272. **Dataset distillation.**<br>
*Wang, Tongzhou and Zhu, Jun-Yan and Torralba, Antonio and Efros, Alexei A.*<br>
arXiv 2018. [[Paper](https://arxiv.org/abs/1811.10959)]
273. **A comprehensive survey to dataset distillation.**<br>
*Lei, Shiye and Tao, Dacheng.*<br>
arXiv 2023. [[Paper](https://arxiv.org/abs/2301.05603)]
274. **Data distillation: A survey.**<br>
*Sachdeva, Noveen and McAuley, Julian.*<br>
TMLR 2023. [[Paper](https://openreview.net/pdf?id=lmXMXP74TO)]
275. **Dataset Condensation with Gradient Matching.**<br>
*Zhao, Bo and Mopuri, Konda Reddy and Bilen, Hakan.*<br>
ICLR 2021. [[Paper](https://openreview.net/forum?id=mSAKhLYLSsl&continueFlag=634046c11e178a0606b18ce2de87a924)]
276. **Remember the past: Distilling datasets into addressable memories for neural networks.**<br>
*Deng, Zhiwei and Russakovsky, Olga.*<br>
NeurIPS 2022. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/de3d2bb604cfc43c81edd2a31b257f03-Paper-Conference.pdf)]
277. **Efficient dataset distillation using random feature approximation.**<br>
*Loo, Noel and Hasani, Ramin and Amini, Alexander and Rus, Daniela.*<br>
NeurIPS 2022. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/5a28f46993c19f428f482cc59db40870-Paper-Conference.pdf)]
278. **Dataset distillation using neural feature regression.**<br>
*Zhou, Yongchao and Nezhadarya, Ehsan and Ba, Jimmy.*<br>
NeurIPS 2022. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/3fe2a777282299ecb4f9e7ebb531f0ab-Paper-Conference.pdf)]
279. **Dataset condensation with differentiable siamese augmentation.**<br>
*Zhao, Bo and Bilen, Hakan.*<br>
ICML 2021. [[Paper](http://proceedings.mlr.press/v139/zhao21a/zhao21a.pdf)]
280. **Dataset condensation via efficient synthetic-data parameterization.**<br>
*Kim, Jang-Hyun and Kim, Jinuk and Oh, Seong Joon and Yun, Sangdoo and Song, Hwanjun and Jeong, Joonhyun and Ha, Jung-Woo and Song, Hyun Oh.*<br>
ICML 2022. [[Paper](https://proceedings.mlr.press/v162/kim22c/kim22c.pdf)]
281. **Dataset distillation by matching training trajectories.**<br>
*Cazenavette, George and Wang, Tongzhou and Torralba, Antonio and Efros, Alexei A and Zhu, Jun-Yan.*<br>
CVPR 2022. [[Paper](https://openaccess.thecvf.com/content/CVPR2022W/VDU/papers/Cazenavette_Dataset_Distillation_by_Matching_Training_Trajectories_CVPRW_2022_paper.pdf)]
282. **Scaling up dataset distillation to imagenet-1k with constant memory.**<br>
*Cui, Justin and Wang, Ruochen and Si, Si and Hsieh, Cho-Jui.*<br>
ICML 2023. [[Paper](https://proceedings.mlr.press/v202/cui23e/cui23e.pdf)]
283. **Dataset condensation with distribution matching.**<br>
*Zhao, Bo and Bilen, Hakan.*<br>
WACV 2023. [[Paper](https://openaccess.thecvf.com/content/WACV2023/papers/Zhao_Dataset_Condensation_With_Distribution_Matching_WACV_2023_paper.pdf)]
284. **Cafe: Learning to condense dataset by aligning features.**<br>
*Wang, Kai and Zhao, Bo and Peng, Xiangyu and Zhu, Zheng and Yang, Shuo and Wang, Shuo and Huang, Guan and Bilen, Hakan and Wang, Xinchao and You, Yang.*<br>
CVPR 2022. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_CAFE_Learning_To_Condense_Dataset_by_Aligning_Features_CVPR_2022_paper.pdf)]
285. **Probabilistic bilevel coreset selection.**<br>
*Zhou, Xiao and Pi, Renjie and Zhang, Weizhong and Lin, Yong and Chen, Zonghao and Zhang, Tong.*<br>
ICML 2022. [[Paper](https://proceedings.mlr.press/v162/zhou22h/zhou22h.pdf)]
286. **Synthesizing Informative Training Samples with GAN.**<br>
*Zhao, Bo and Bilen, Hakan.*<br>
NeurIPS Workshop 2022. [[Paper](https://openreview.net/pdf?id=frAv0jtUMfS)]
287. **Beyond neural scaling laws: beating power law scaling via data pruning.**<br>
*Sorscher, Ben and Geirhos, Robert and Shekhar, Shashank and Ganguli, Surya and Morcos, Ari.*<br>
NeurIPS 2022. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/7b75da9b61eda40fa35453ee5d077df6-Paper-Conference.pdf)]
288. **InfoBatch: Lossless Training Speed Up by Unbiased Dynamic Data Pruning.**<br>
*Qin, Ziheng and Wang, Kai and Zheng, Zangwei and Gu, Jianyang and Peng, Xiangyu and Zhou, Daquan and You, Yang.*<br>
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.04947)]
289. **Dataset distillation via factorization.**<br>
*Liu, Songhua and Wang, Kai and Yang, Xingyi and Ye, Jingwen and Wang, Xinchao.*<br>
NeurIPS 2022. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/07bc722f08f096e6ea7ee99349ff0a86-Paper-Conference.pdf)]
### Subset selection
290. **A Survey of Data Optimization for Problems in Computer Vision Datasets.**<br>
*Wan, Zhijing and Wang, Zhixiang and Chung, CheukTing and Wang, Zheng.*<br>
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.11717)]
291. **Trivial or Impossible---dichotomous data difficulty masks model differences (on ImageNet and beyond).**<br>
*Meding, Kristof and Buschoff, Luca M Schulze and Geirhos, Robert and Wichmann, Felix A.*<br>
ICLR 2022. [[Paper](https://openreview.net/pdf?id=C_vsGwEIjAr)]
292. **What neural networks memorize and why: Discovering the long tail via influence estimation.**<br>
*Feldman, Vitaly and Zhang, Chiyuan.*<br>
NeurIPS 2020. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/1e14bfe2714193e7af5abc64ecbd6b46-Paper.pdf)]
293. **Semantic Redundancies in Image-Classification Datasets: The 10% You Don't Need.**<br>
*Birodkar, Vighnesh and Mobahi, Hossein and Bengio, Samy.*<br>
arXiv 2019. [[Paper](https://arxiv.org/abs/1901.11409)]
294. **Learning with confident examples: Rank pruning for robust classification with noisy labels.**<br>
*Northcutt, Curtis G and Wu, Tailin and Chuang, Isaac L.*<br>
arXiv 2017. [[Paper](https://arxiv.org/abs/1705.01936)]
295. **Learning from less data: A unified data subset selection and active learning framework for computer vision.**<br>
*Kaushal, Vishal and Iyer, Rishabh and Kothawade, Suraj and Mahadev, Rohan and Doctor, Khoshrav and Ramakrishnan, Ganesh.*<br>
WACV 2019. [[Paper](https://ieeexplore.ieee.org/abstract/document/8658965)]
296. **Towards Sustainable Learning: Coresets for Data-efficient Deep Learning.**<br>
*Yang, Yu and Kang, Hao and Mirzasoleiman, Baharan.*<br>
ICML 2023. [[Paper](https://openreview.net/pdf?id=ASOCqTnWIY)]
297. **Coresets for data-efficient training of machine learning models.**<br>
*Mirzasoleiman, Baharan and Bilmes, Jeff and Leskovec, Jure.*<br>
ICML 2020. [[Paper](https://proceedings.mlr.press/v119/mirzasoleiman20a/mirzasoleiman20a.pdf)]
## Other typical techniques
### Pure mathematical optimization
298. **DivAug: Plug-in automated data augmentation with explicit diversity maximization.**<br>
*Liu, Zirui and Jin, Haifeng and Wang, Ting-Hsiang and Zhou, Kaixiong and Hu, Xia.*<br>
ICCV 2021. [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_DivAug_Plug-In_Automated_Data_Augmentation_With_Explicit_Diversity_Maximization_ICCV_2021_paper.pdf)]
299. **Submodular batch selection for training deep neural networks.**<br>
*Joseph, KJ and Singh, Krishnakant and Balasubramanian, Vineeth N.*<br>
IJCAI 2019. [[Paper](https://dl.acm.org/doi/abs/10.5555/3367243.3367412)]
300. **Submodular Meta Data Compiling for Meta Optimization.**<br>
*Su, Fengguang and Zhu, Yu and Wu, Ou and Deng, Yingjun.*<br>
ECML/PKDD 2022. [[Paper](https://2022.ecmlpkdd.org/wp-content/uploads/2022/09/sub_474.pdf)]
301. **Regularization via structural label smoothing.**<br>
*Li, Weizhi and Dasarathy, Gautam and Berisha, Visar.*<br>
AISTATS 2020. [[Paper](https://proceedings.mlr.press/v108/li20e/li20e.pdf)]
302. **Generalized Entropy Regularization or: There’s Nothing Special about Label Smoothing.**<br>
*Meister, Clara and Salesky, Elizabeth and Cotterell, Ryan.*<br>
ACL 2020. [[Paper](https://aclanthology.org/2020.acl-main.615/)]
303. **Fairness with adaptive weights.**<br>
*Chai, Junyi and Wang, Xiaoqian.*<br>
ICML 2022. [[Paper](https://proceedings.mlr.press/v162/chai22a/chai22a.pdf)]
304. **Multi-label adversarial perturbations.**<br>
*Song, Qingquan and Jin, Haifeng and Huang, Xiao and Hu, Xia.*<br>
ICDM 2018. [[Paper](https://ieeexplore.ieee.org/abstract/document/8594975)]
305. **Tkml-ap: Adversarial attacks to top-k multi-label learning.**<br>
*Hu, Shu and Ke, Lipeng and Wang, Xin and Lyu, Siwei.*<br>
ICCV 2021. [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Hu_TkML-AP_Adversarial_Attacks_to_Top-k_Multi-Label_Learning_ICCV_2021_paper.pdf)]
306. **Evolutionary Multi-Label Adversarial Examples: An Effective Black-Box Attack.**<br>
*Kong, Linghao and Luo, Wenjian and Zhang, Hongwei and Liu, Yang and Shi, Yuhui.*<br>
IEEE TAI 2022. [[Paper](https://ieeexplore.ieee.org/abstract/document/9857594)]
### Technique combination                                               
307. **Mitigating Exposure Bias in Grammatical Error Correction with Data Augmentation and Reweighting.**<br>
*Cao, Hannan and Yang, Wenmian and Ng, Hwee Tou.*<br>
EACL 2023. [[Paper](https://aclanthology.org/2023.eacl-main.155/)]
308. **Counterfactual data augmentation for neural machine translation.**<br>
*Liu, Qi and Kusner, Matt and Blunsom, Phil.*<br>
NAACL 2021. [[Paper](https://aclanthology.org/2021.naacl-main.18/)]
309. **Fasa: Feature augmentation and sampling adaptation for long-tailed instance segmentation.**<br>
*Zang, Yuhang and Huang, Chen and Loy, Chen Change.*<br>
ICCV 2021. [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zang_FASA_Feature_Augmentation_and_Sampling_Adaptation_for_Long-Tailed_Instance_Segmentation_ICCV_2021_paper.pdf)]
310. **Adaptive logit adjustment loss for long-tailed visual recognition.**<br>
*Zhao, Yan and Chen, Weicong and Tan, Xu and Huang, Kai and Zhu, Jihong.*<br>
AAAI 2022. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/20258)]
311. **Combining Adversaries with Anti-adversaries in Training.**<br>
*Zhou, Xiaoling and Yang, Nan and Wu, Ou.*<br>
AAAI 2023. [[Paper](https://arxiv.org/abs/2304.12550)]
312. **Umix: Improving importance weighting for subpopulation shift via uncertainty-aware mixup.**<br>
*Han, Zongbo and Liang, Zhipeng and Yang, Fan and Liu, Liu and Li, Lanqing and Bian, Yatao and Zhao, Peilin and Wu, Bingzhe and Zhang, Changqing and Yao, Jianhua.*<br>
NeurIPS 2022. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/f593c9c251d4d7cf14d4ab9861dfb7eb-Paper-Conference.pdf)]
313. **Imagine by reasoning: A reasoning-based implicit semantic data augmentation for long-tailed classification.**<br>
*Chen, Xiaohua and Zhou, Yucan and Wu, Dayan and Zhang, Wanqian and Zhou, Yu and Li, Bo and Wang, Weiping.*<br>
AAAI 2022. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/19912)]
314. **Focal loss for dense object detection.**<br>
*Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Dollár, Piotr.*<br>
ICCV 2017. [[Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)]                                                                
# Data optimization theories
## Formalization           
315. **Why does rebalancing class-unbalanced data improve AUC for linear discriminant analysis?**<br>
*Xue, Jing-Hao and Hall, Peter.*<br>
TPAMI 2015. [[Paper](https://ieeexplore.ieee.org/abstract/document/6906278)]                                  
316. **Adversarial examples are not bugs, they are features.**<br>
*Ilyas, Andrew and Santurkar, Shibani and Tsipras, Dimitris and Engstrom, Logan and Tran, Brandon and Madry, Aleksander.*<br>
NeurIPS 2019. [[Paper](https://proceedings.neurips.cc/paper/2019/file/e2c420d928d4bf8ce0ff2ec19b371514-Paper.pdf)]      
317. **A theoretical distribution analysis of synthetic minority oversampling technique (SMOTE) for imbalanced learning.**<br>
*Elreedy, Dina and Atiya, Amir F and Kamalov, Firuz.*<br>
Machine Learning 2023. [[Paper](https://link.springer.com/article/10.1007/s10994-022-06296-4)]   
318. **FAIR: Fair adversarial instance re-weighting.**<br>
*Petrović, Andrija and Nikolić, Mladen and Radovanović, Sandro and Delibašić, Boris and Jovanović, Miloš.*<br>
Neurocomputing 2022. [[Paper](https://www.sciencedirect.com/science/article/pii/S0925231221019408?casa_token=zl0smR7i06AAAAAA:ybSefSP57QrNHVMLB9lb4rTQLCubIPA2Ggnh87bSC3Dv4faAC4f2zg5a38HQwA-6OyDUVpIK4C4)]
319. **Delving into deep imbalanced regression.**<br>
*Yang, Yuzhe and Zha, Kaiwen and Chen, Yingcong and Wang, Hao and Katabi, Dina.*<br>
ICML 2021. [[Paper](https://proceedings.mlr.press/v139/yang21m/yang21m.pdf)]  
320. **Long-tailed visual recognition via gaussian clouded logit adjustment.**<br>
*Li, Mengke and Cheung, Yiu-ming and Lu, Yang.*<br>
CVPR 2022. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Long-Tailed_Visual_Recognition_via_Gaussian_Clouded_Logit_Adjustment_CVPR_2022_paper.pdf)] 
321. **Balanced mse for imbalanced visual regression.**<br>
*Ren, Jiawei and Zhang, Mingyuan and Yu, Cunjun and Liu, Ziwei.*<br>
CVPR 2022. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Ren_Balanced_MSE_for_Imbalanced_Visual_Regression_CVPR_2022_paper.pdf)] 
322. **Gaussian distribution based oversampling for imbalanced data classification.**<br>
*Xie, Yuxi and Qiu, Min and Zhang, Haibo and Peng, Lizhi and Chen, Zhenxiang.*<br>
TKDE 2022. [[Paper](https://ieeexplore.ieee.org/abstract/document/9063462)] 
323. **Rethinking the value of labels for improving class-imbalanced learning.**<br>
*Yang, Yuzhe and Xu, Zhi.*<br>
NeurIPS 2020. [[Paper](https://proceedings.neurips.cc/paper/2020/file/e025b6279c1b88d3ec0eca6fcb6e6280-Paper.pdf)] 
324. **Self-supervised Learning is More Robust to Dataset Imbalance.**<br>
*Liu, Hong and HaoChen, Jeff Z and Gaidon, Adrien and Ma, Tengyu.*<br>
NeurIPS Workshop 2021. [[Paper](https://openreview.net/pdf?id=vUz4JPRLpGx)] 
325. **Learning imbalanced datasets with label-distribution-aware margin loss.**<br>
*Cao, Kaidi and Wei, Colin and Gaidon, Adrien and Arechiga, Nikos and Ma, Tengyu.*<br>
NeurIPS 2019. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/621461af90cadfdaf0e8d4cc25129f91-Paper.pdf)]
326. **An Optimal Transport View of Class-Imbalanced Visual Recognition.**<br>
*Jin, Lianbao and Lang, Dayu and Lei, Na.*<br>
IJCV 2023. [[Paper](https://link.springer.com/article/10.1007/s11263-023-01831-9)] 
327. **Long-tail learning via logit adjustment.**<br>
*Menon, Aditya Krishna and Jayasumana, Sadeep and Rawat, Ankit Singh and Jain, Himanshu and Veit, Andreas and Kumar, Sanjiv.*<br>
ICLR 2021. [[Paper](https://openreview.net/pdf?id=37nvvqkCo5)]
328. **Imbalanced deep learning by minority class incremental rectification.**<br>
*Dong, Qi and Gong, Shaogang and Zhu, Xiatian.*<br>
TPAMI 2019. [[Paper](https://ieeexplore.ieee.org/abstract/document/8353718)]
329. **Understanding the role of importance weighting for deep learning.**<br>
*Xu, Da and Ye, Yuting and Ruan, Chuanwei.*<br>
arXiv 2021. [[Paper](https://arxiv.org/abs/2103.15209)]  
330. **Is Importance Weighting Incompatible with Interpolating Classifiers?**<br>
*Wang, Ke Alexander and Chatterji, Niladri Shekhar and Haque, Saminul and Hashimoto, Tatsunori.*<br>
ICLR 2022. [[Paper](https://openreview.net/pdf?id=uqBOne3LUKy)] 
331. **A theoretical analysis on independence-driven importance weighting for covariate-shift generalization.**<br>
*Xu, Renzhe and Zhang, Xingxuan and Shen, Zheyan and Zhang, Tong and Cui, Peng.*<br>
ICML 2022. [[Paper](https://proceedings.mlr.press/v162/xu22o/xu22o.pdf)] 
332. **Zero-Shot Logit Adjustment.**<br>
*Chen, Dubing and Shen, Yuming and Zhang, Haofeng and Torr, Philip HS.*<br>
IJCAI 2022. [[Paper](https://www.ijcai.org/proceedings/2022/0114.pdf)] 
333. **Bias Mimicking: A Simple Sampling Approach for Bias Mitigation.**<br>
*Qraitem, Maan and Saenko, Kate and Plummer, Bryan A.*<br>
CVPR 2023. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Qraitem_Bias_Mimicking_A_Simple_Sampling_Approach_for_Bias_Mitigation_CVPR_2023_paper.pdf)]
334. **Sample selection for fair and robust training.**<br>
*Roh, Yuji and Lee, Kangwook and Whang, Steven and Suh, Changho.*<br>
NeurIPS 2021. [[Paper](https://proceedings.neurips.cc/paper/2021/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf)] 
335. **Boosting causal discovery via adaptive sample reweighting.**<br>
*Zhang, An and Liu, Fangfu and Ma, Wenchang and Cai, Zhibo and Wang, Xiang and Chua, Tat-Seng.*<br>
ICLR 2023. [[Paper](https://openreview.net/pdf?id=LNpMtk15AS4)] 
336. **Adversarial defense via learning to generate diverse attacks.**<br>
*Jang, Yunseok and Zhao, Tianchen and Hong, Seunghoon and Lee, Honglak.*<br>
ICCV 2019. [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Jang_Adversarial_Defense_via_Learning_to_Generate_Diverse_Attacks_ICCV_2019_paper.pdf)]
337. **Automatic data augmentation via invariance-constrained learning.**<br>
*Hounie, Ignacio and Chamon, Luiz FO and Ribeiro, Alejandro.*<br>
ICML 2023. [[Paper](https://proceedings.mlr.press/v202/hounie23a/hounie23a.pdf)] 
338. **Recovering from biased data: Can fairness constraints improve accuracy?**<br>
*Blum, Avrim and Stangl, Kevin.*<br>
FORC 2020. [[Paper](https://par.nsf.gov/biblio/10190440)]                               
## Explanation
339. **A theoretical analysis of catastrophic forgetting through the ntk overlap matrix.**<br>
*Doan, Thang and Bennani, Mehdi Abbana and Mazoure, Bogdan and Rabusseau, Guillaume and Alquier, Pierre.*<br>
AISTATS 2021. [[Paper](https://proceedings.mlr.press/v130/doan21a/doan21a.pdf)] 
340. **On the generalization mystery in deep learning.**<br>
*Chatterjee, Satrajit and Zielinski, Piotr.*<br>
arXiv 2022. [[Paper](https://arxiv.org/abs/2203.10036)] 
341. **Not all samples are created equal: Deep learning with importance sampling.**<br>
*Katharopoulos, Angelos and Fleuret, François.*<br>
ICML 2018. [[Paper](https://proceedings.mlr.press/v80/katharopoulos18a/katharopoulos18a.pdf)]  
342. **Biased importance sampling for deep neural network training.**<br>
*Katharopoulos, Angelos and Fleuret, François.*<br>
arXiv 2017. [[Paper](https://arxiv.org/abs/1706.00043)] 
343. **Less is better: Unweighted data subsampling via influence function.**<br>
*Wang, Zifeng and Zhu, Hong and Dong, Zhenhua and He, Xiuqiang and Huang, Shao-Lun.*<br>
AAAI 2020. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/6103)]
344. **A kernel theory of modern data augmentation.**<br>
*Dao, Tri and Gu, Albert and Ratner, Alexander and Smith, Virginia and De Sa, Chris and Ré, Christopher.*<br>
ICML 2019. [[Paper](https://proceedings.mlr.press/v97/dao19b/dao19b.pdf)]
345. **Maximum-entropy adversarial data augmentation for improved generalization and robustness.**<br>
*Zhao, Long and Liu, Ting and Peng, Xi and Metaxas, Dimitris.*<br>
NeurIPS 2020. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/a5bfc9e07964f8dddeb95fc584cd965d-Paper.pdf)]
346. **A Unified Framework for Adversarial Attacks on Multi-Source Domain Adaptation.**<br>
*Wu, Jun and He, Jingrui.*<br>
TKDE 2022. [[Paper](https://ieeexplore.ieee.org/abstract/document/9994047)] 
347. **Adversarial examples are a natural consequence of test error in noise.**<br>
*Gilmer, Justin and Ford, Nicolas and Carlini, Nicholas and Cubuk, Ekin.*<br>
ICML 2019. [[Paper](https://proceedings.mlr.press/v97/gilmer19a/gilmer19a.pdf)] 
348. **Improved ood generalization via adversarial training and pretraing.**<br>
*Yi, Mingyang and Hou, Lu and Sun, Jiacheng and Shang, Lifeng and Jiang, Xin and Liu, Qun and Ma, Zhiming.*<br>
ICML 2021. [[Paper](https://proceedings.mlr.press/v139/yi21a/yi21a.pdf)] 
349. **Lower bounds on the robustness to adversarial perturbations.**<br>
*Peck, Jonathan and Roels, Joris and Goossens, Bart and Saeys, Yvan.*<br>
NeurIPS 2017. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/298f95e1bf9136124592c8d4825a06fc-Paper.pdf)] 
350. **Towards understanding label smoothing.**<br>
*Xu, Yi and Xu, Yuanhong and Qian, Qi and Li, Hao and Jin, Rong.*<br>
arXiv 2020. [[Paper](https://arxiv.org/abs/2006.11653)] 
351. **Class-Level Logit Perturbation.**<br>
*Li, Mengyang and Su, Fengguang and Wu, Ou and Zhang, Ji.*<br>
TNNLS 2023. [[Paper](https://ieeexplore.ieee.org/abstract/document/10130785/)]
352. **What is the effect of importance weighting in deep learning?**<br>
*Byrd, Jonathon and Lipton, Zachary.*<br>
ICML 2019. [[Paper](https://proceedings.mlr.press/v97/byrd19a/byrd19a.pdf)]    
353. **Rethinking importance weighting for deep learning under distribution shift.**<br>
*Fang, Tongtong and Lu, Nan and Niu, Gang and Sugiyama, Masashi.*<br>
NeurIPS 2020. [[Paper](https://proceedings.neurips.cc/paper/2020/file/8b9e7ab295e87570551db122a04c6f7c-Paper.pdf)]
354. **A theoretical understanding of self-paced learning.**<br>
*Meng, Deyu and Zhao, Qian and Jiang, Lu.*<br>
Information Sciences 2017. [[Paper](https://www.sciencedirect.com/science/article/pii/S0020025517307521?casa_token=wYHAnjKLNvIAAAAA:X_tCLrum-44D5Y0KIn1GtdW_MQaD7iYzQe0kBVA2NSH-WNEIOZ89Q4DB4Kncz0nuvwq6NRUMA3c)] 
355. **Curriculum learning by transfer learning: Theory and experiments with deep networks.**<br>
*Weinshall, Daphna and Cohen, Gad and Amir, Dan.*<br>
ICML 2018. [[Paper](https://proceedings.mlr.press/v80/weinshall18a/weinshall18a.pdf)] 
356. **Rethinking data distillation: Do not overlook calibration.**<br>
*Zhu, Dongyao and Lei, Bowen and Zhang, Jie and Fang, Yanbo and Xie, Yiqun and Zhang, Ruqi and Xu, Dongkuan.*<br>
ICCV 2023. [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhu_Rethinking_Data_Distillation_Do_Not_Overlook_Calibration_ICCV_2023_paper.pdf)] 
357. **Privacy for free: How does dataset condensation help privacy?**<br>
*Dong, Tian and Zhao, Bo and Lyu, Lingjuan.*<br>
ICML 2022. [[Paper](https://proceedings.mlr.press/v162/dong22c/dong22c.pdf)] 
358. **Avoiding overfitting: A survey on regularization methods for convolutional neural networks.**<br>
*Santos, Claudio Filipi Gonçalves Dos and Papa, João Paulo.*<br>
ACM Computing Surveys 2022. [[Paper](https://dl.acm.org/doi/full/10.1145/3510413)]
359. **Implicit semantic data augmentation for deep networks.**<br>
*Wang, Yulin and Pan, Xuran and Song, Shiji and Zhang, Hong and Huang, Gao and Wu, Cheng.*<br>
NeurIPS 2019. [[Paper](https://proceedings.neurips.cc/paper/2019/file/15f99f2165aa8c86c9dface16fefd281-Paper.pdf)]
360. **Revisiting knowledge distillation via label smoothing regularization.**<br>
*Yuan, Li and Tay, Francis EH and Li, Guilin and Wang, Tao and Feng, Jiashi.*<br>
CVPR 2020. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yuan_Revisiting_Knowledge_Distillation_via_Label_Smoothing_Regularization_CVPR_2020_paper.pdf)] 
361. **Neural networks regularization with graph-based local resampling.**<br>
*Assis, Alex D and Torres, Luiz CB and Araújo, Lourenço RG and Hanriot, Vítor M and Braga, Antonio P.*<br>
IEEE Access 2021. [[Paper](https://ieeexplore.ieee.org/abstract/document/9383228)]
# Connections among different techniques
## Connections via data perception  
362. **Probabilistic anchor assignment with iou prediction for object detection.**<br>
*Kim, Kang and Lee, Hee Seok.*<br>
ECCV 2020. [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-58595-2_22)]
363. **Self-paced data augmentation for training neural networks.**<br>
*Takase, Tomoumi and Karakida, Ryo and Asoh, Hideki.*<br>
Neurocomputing 2021. [[Paper](https://www.sciencedirect.com/science/article/pii/S0925231221003374?casa_token=LvoZhgDinyoAAAAA:E-94HYU35zzyae3x7Q2wwvO_UWSsDlMXYkJr3QuBOCB8WVbImWYQrNekPMDNIQQVfKYa9T2qWr4)]
364. **Logit perturbation.**<br>
*Li, Mengyang and Su, Fengguang and Wu, Ou and Zhang, Ji.*<br>
AAAI 2022. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/20024)]
365. **A theoretical understanding of self-paced learning.**<br>
*Meng, Deyu and Zhao, Qian and Jiang, Lu.*<br>
Information Sciences 2017. [[Paper](https://www.sciencedirect.com/science/article/pii/S0020025517307521?casa_token=wYHAnjKLNvIAAAAA:X_tCLrum-44D5Y0KIn1GtdW_MQaD7iYzQe0kBVA2NSH-WNEIOZ89Q4DB4Kncz0nuvwq6NRUMA3c)] 
366. **Metadata Archaeology: Unearthing Data Subsets by Leveraging Training Dynamics.**<br>
*Siddiqui, Shoaib Ahmed and Rajkumar, Nitarshan and Maharaj, Tegan and Krueger, David and Hooker, Sara.*<br>
ICLR 2023. [[Paper](https://openreview.net/pdf?id=PvLnIaJbt9)] 
367. **Not all samples are created equal: Deep learning with importance sampling.**<br>
*Katharopoulos, Angelos and Fleuret, François.*<br>
ICML 2018. [[Paper](https://proceedings.mlr.press/v80/katharopoulos18a/katharopoulos18a.pdf)]  
368. **MixGradient: A gradient-based re-weighting scheme with mixup for imbalanced data streams.**<br>
*Peng, Xinyu and Wang, Fei-Yue and Li, Li.*<br>
Neural Networks 2023. [[Paper](https://www.sciencedirect.com/science/article/pii/S0893608023000801?casa_token=BuyZi4hmHDgAAAAA:dmJKzhUabuhWUE4l4rSocLbJB2iS-l5qws2dLK45Db_P1r4pDvpLI4UW8AobBPMDtWMnN_WAPHU)] 
369. **Combining Adversaries with Anti-adversaries in Training.**<br>
*Zhou, Xiaoling and Yang, Nan and Wu, Ou.*<br>
AAAI 2023. [[Paper](https://arxiv.org/abs/2304.12550)]
370. **Gradient harmonized single-stage detector.**<br>
*Li, Buyu and Liu, Yu and Wang, Xiaogang.*<br>
AAAI 2019. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/4877)]
371. **Dataset Condensation with Gradient Matching.**<br>
*Zhao, Bo and Mopuri, Konda Reddy and Bilen, Hakan.*<br>
ICLR 2021. [[Paper](https://openreview.net/forum?id=mSAKhLYLSsl&continueFlag=634046c11e178a0606b18ce2de87a924)]                                   
## Connections via application scenarios
372. **Resampling-based noise correction for crowdsourcing.**<br>
*Xu, Wenqiang and Jiang, Liangxiao and Li, Chaoqun.*<br>
Journal of Experimental & Theoretical Artificial Intelligence 2021. [[Paper](https://www.tandfonline.com/doi/abs/10.1080/0952813X.2020.1806519)] 
373. **Generative dataset distillation.**<br>
*Huang, Chengeng and Zhang, Sihai.*<br>
BigCom 2021. [[Paper](https://ieeexplore.ieee.org/abstract/document/9546880)] 
374. **Towards Understanding Deep Learning from Noisy Labels with Small-Loss Criterion.**<br>
*Gui, Xian-Jin and Wang, Wei and Tian, Zhang-Hao.*<br>
IJCAI 2021. [[Paper](https://www.ijcai.org/proceedings/2021/0340.pdf)]  
375. **Data valuation using reinforcement learning.**<br>
*Yoon, Jinsung and Arik, Sercan and Pfister, Tomas.*<br>
ICML 2020. [[Paper](https://proceedings.mlr.press/v119/yoon20a/yoon20a.pdf)] 
376. **Less is better: Unweighted data subsampling via influence function.**<br>
*Wang, Zifeng and Zhu, Hong and Dong, Zhenhua and He, Xiuqiang and Huang, Shao-Lun.*<br>
AAAI 2020. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/6103)] 
377. **Augmentation strategies for learning with noisy labels.**<br>
*Nishi, Kento and Ding, Yi and Rich, Alex and Hollerer, Tobias.*<br>
CVPR 2021. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Nishi_Augmentation_Strategies_for_Learning_With_Noisy_Labels_CVPR_2021_paper.pdf)]
378. **PropMix: Hard Sample Filtering and Proportional MixUp for Learning with Noisy Labels.**<br>
*Cordeiro, Filipe R and Belagiannis, Vasileios and Reid, Ian and Carneiro, Gustavo.*<br>
BMVC 2021. [[Paper](https://www.bmvc2021-virtualconference.com/assets/papers/0908.pdf)] 
379. **Adversarial Auto-Augment with Label Preservation: A Representation Learning Principle Guided Approach.**<br>
*Yang, Kaiwen and Sun, Yanchao and Su, Jiahao and He, Fengxiang and Tian, Xinmei and Huang, Furong and Zhou, Tianyi and Tao, Dacheng.*<br>
NeurIPS 2022. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/8a1c4a54d73728d4d61701e320687c6d-Paper-Conference.pdf)] 
380. **DivideMix: Learning with Noisy Labels as Semi-supervised Learning.**<br>
*Li, Junnan and Socher, Richard and Hoi, Steven CH.*<br>
ICLR 2020. [[Paper](https://openreview.net/pdf?id=HJgExaVtwr)] 
381. **Rethinking the inception architecture for computer vision.**<br>
*Szegedy, Christian and Vanhoucke, Vincent and Ioffe, Sergey and Shlens, Jon and Wojna, Zbigniew.*<br>
CVPR 2016. [[Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)]
382. **Training deep neural networks on noisy labels with bootstrapping.**<br>
*Reed, Scott and Lee, Honglak and Anguelov, Dragomir and Szegedy, Christian and Erhan, Dumitru and Rabinovich, Andrew.*<br>
ICLR workshop 2015. [[Paper](https://imgtec.eetrend.com/sites/imgtec.eetrend.com/files/201709/blog/10381-29627-deep.pdf)]
383. **Self-paced learning for latent variable models.**<br>
*Kumar, M and Packer, Benjamin and Koller, Daphne.*<br>
NeurIPS 2010. [[Paper](https://proceedings.neurips.cc/paper/2010/file/e57c6b956a6521b28495f2886ca0977a-Paper.pdf)]
384. **Cmw-net: Learning a class-aware sample weighting mapping for robust deep learning.**<br>
*Shu, Jun and Yuan, Xiang and Meng, Deyu and Xu, Zongben.*<br>
TPAMI 2023. [[Paper](https://ieeexplore.ieee.org/abstract/document/10113668)] 
385. **Learning fast sample re-weighting without reward data.**<br>
*Zhang, Zizhao and Pfister, Tomas.*<br>
ICCV 2021. [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Learning_Fast_Sample_Re-Weighting_Without_Reward_Data_ICCV_2021_paper.pdf)] 
386. **Derivative manipulation for general example weighting.**<br>
*Wang, Xinshao and Kodirov, Elyor and Hua, Yang and Robertson, Neil M.*<br>
arXiv 2019. [[Paper](https://arxiv.org/abs/1905.11233)] 
387. **Distillhash: Unsupervised deep hashing by distilling data pairs.**<br>
*Yang, Erkun and Liu, Tongliang and Deng, Cheng and Liu, Wei and Tao, Dacheng.*<br>
CVPR 2019. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yang_DistillHash_Unsupervised_Deep_Hashing_by_Distilling_Data_Pairs_CVPR_2019_paper.pdf)] 
388. **Coresets for robust training of deep neural networks against noisy labels.**<br>
*Mirzasoleiman, Baharan and Cao, Kaidi and Leskovec, Jure.*<br>
NeurIPS 2020. [[Paper](https://proceedings.neurips.cc/paper/2020/file/8493eeaccb772c0878f99d60a0bd2bb3-Paper.pdf)] 
389. **Prioritized training on points that are learnable, worth learning, and not yet learnt.**<br>
*Mindermann, Sören and Brauner, Jan M and Razzak, Muhammed T and Sharma, Mrinank and Kirsch, Andreas and Xu, Winnie and Höltgen, Benedikt and Gomez, Aidan N and Morisot, Adrien and Farquhar, Sebastian and Gal, Yarin.*<br>
ICML 2022. [[Paper](https://proceedings.mlr.press/v162/mindermann22a/mindermann22a.pdf)] 
390. **Background data resampling for outlier-aware classification.**<br>
*Li, Yi and Vasconcelos, Nuno.*<br>
CVPR 2020. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Background_Data_Resampling_for_Outlier-Aware_Classification_CVPR_2020_paper.pdf)]  
391. **Robust semi-supervised classification based on data augmented online ELMs with deep features.**<br>
*Hu, Xiaochang and Zeng, Yujun and Xu, Xin and Zhou, Sihang and Liu, Li.*<br>
Knowledge-Based Systems 2021. [[Paper](https://www.sciencedirect.com/science/article/pii/S0950705121005694?casa_token=qAulNIchwkQAAAAA:JAtV11ozlcvF2yI0CTywgKC6uKoetfmS162H0uuPMoOKI-ZrS8JS7ggy5plYixgrf0SzmjgAYcU)] 
392. **Combining Adversaries with Anti-adversaries in Training.**<br>
*Zhou, Xiaoling and Yang, Nan and Wu, Ou.*<br>
AAAI 2023. [[Paper](https://arxiv.org/abs/2304.12550)]
393. **Cafe: Learning to condense dataset by aligning features.**<br>
*Wang, Kai and Zhao, Bo and Peng, Xiangyu and Zhu, Zheng and Yang, Shuo and Wang, Shuo and Huang, Guan and Bilen, Hakan and Wang, Xinchao and You, Yang.*<br>
CVPR 2022. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_CAFE_Learning_To_Condense_Dataset_by_Aligning_Features_CVPR_2022_paper.pdf)]
394. **Coreset Sampling from Open-Set for Fine-Grained Self-Supervised Learning.**<br>
*Kim, Sungnyun and Bae, Sangmin and Yun, Se-Young.*<br>
CVPR 2023. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Kim_Coreset_Sampling_From_Open-Set_for_Fine-Grained_Self-Supervised_Learning_CVPR_2023_paper.pdf)] 
395. **Learning with neighbor consistency for noisy labels.**<br>
*Iscen, Ahmet and Valmadre, Jack and Arnab, Anurag and Schmid, Cordelia.*<br>
CVPR 2022. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Iscen_Learning_With_Neighbor_Consistency_for_Noisy_Labels_CVPR_2022_paper.pdf)]
396. **Learning multiple layers of features from tiny images.**<br>
*Krizhevsky, Alex.*<br>
Technical report 2009. [[Paper](https://www.cs.utoronto.ca/~kriz/learning-features-2009-TR.pdf)]
397. **Learning from massive noisy labeled data for image classification.**<br>
*Xiao, Tong and Xia, Tian and Yang, Yi and Huang, Chang and Wang, Xiaogang.*<br>
CVPR 2015. [[Paper](https://openaccess.thecvf.com/content_cvpr_2015/papers/Xiao_Learning_From_Massive_2015_CVPR_paper.pdf)]
398. **Reading digits in natural images with unsupervised feature learning.**<br>
*Netzer, Yuval and Wang, Tao and Coates, Adam and Bissacco, Alessandro and Wu, Bo and Ng, Andrew Y.*<br>
NeurIPSW 2011. [[Paper](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/37648.pdf)]
399. **Webvision database: Visual learning and understanding from web data.**<br>
*Li, Wen and Wang, Limin and Li, Wei and Agustsson, Eirikur and Van Gool, Luc.*<br>
arXiv 2017. [[Paper](https://arxiv.org/abs/1708.02862)]
400. **Remix: Calibrated resampling for class imbalance in deep learning.**<br>
*Bellinger, Colin and Corizzo, Roberto and Japkowicz, Nathalie.*<br>
arXiv 2020. [[Paper](https://arxiv.org/abs/2012.02312)]
401. **Global and Local Mixture Consistency Cumulative Learning for Long-tailed Visual Recognitions.**<br>
*Du, Fei and Yang, Peng and Jia, Qi and Nan, Fengtao and Chen, Xiaoting and Yang, Yun.*<br>
CVPR 2023. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Du_Global_and_Local_Mixture_Consistency_Cumulative_Learning_for_Long-Tailed_Visual_CVPR_2023_paper.pdf)]
402. **Long-tail learning via logit adjustment.**<br>
*Menon, Aditya Krishna and Jayasumana, Sadeep and Rawat, Ankit Singh and Jain, Himanshu and Veit, Andreas and Kumar, Sanjiv.*<br>
ICLR 2021. [[Paper](https://openreview.net/pdf?id=37nvvqkCo5)]
403. **Metasaug: Meta semantic augmentation for long-tailed visual recognition.**<br>
*Li, Shuang and Gong, Kaixiong and Liu, Chi Harold and Wang, Yulin and Qiao, Feng and Cheng, Xinjing.*<br>
CVPR 2021. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_MetaSAug_Meta_Semantic_Augmentation_for_Long-Tailed_Visual_Recognition_CVPR_2021_paper.pdf)]
404. **Focal loss for dense object detection.**<br>
*Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Dollár, Piotr.*<br>
ICCV 2017. [[Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)]    
405. **Class-balanced loss based on effective number of samples.**<br>
*Cui, Yin and Jia, Menglin and Lin, Tsung-Yi and Song, Yang and Belongie, Serge.*<br>
CVPR 2019. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf)]
406. **Submodular Meta Data Compiling for Meta Optimization.**<br>
*Su, Fengguang and Zhu, Yu and Wu, Ou and Deng, Yingjun.*<br>
ECML/PKDD 2022. [[Paper](https://2022.ecmlpkdd.org/wp-content/uploads/2022/09/sub_474.pdf)]
407. **Adaptive second order coresets for data-efficient machine learning.**<br>
*Pooladzandi, Omead and Davini, David and Mirzasoleiman, Baharan.*<br>
ICML 2022. [[Paper](https://proceedings.mlr.press/v162/pooladzandi22a/pooladzandi22a.pdf)]
408. **Learning imbalanced datasets with label-distribution-aware margin loss.**<br>
*Cao, Kaidi and Wei, Colin and Gaidon, Adrien and Arechiga, Nikos and Ma, Tengyu.*<br>
NeurIPS 2019. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/621461af90cadfdaf0e8d4cc25129f91-Paper.pdf)]
409. **Imagine by reasoning: A reasoning-based implicit semantic data augmentation for long-tailed classification.**<br>
*Chen, Xiaohua and Zhou, Yucan and Wu, Dayan and Zhang, Wanqian and Zhou, Yu and Li, Bo and Wang, Weiping.*<br>
AAAI 2022. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/19912)]
410. **Improved distribution matching for dataset condensation.**<br>
*Zhao, Ganlong and Li, Guanbin and Qin, Yipeng and Yu, Yizhou.*<br>
CVPR 2023. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_Improved_Distribution_Matching_for_Dataset_Condensation_CVPR_2023_paper.pdf)]
411. **The inaturalist species classification and detection dataset.**<br>
*Van Horn, Grant and Mac Aodha, Oisin and Song, Yang and Cui, Yin and Sun, Chen and Shepard, Alex and Adam, Hartwig and Perona, Pietro and Belongie, Serge.*<br>
CVPR 2018. [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Van_Horn_The_INaturalist_Species_CVPR_2018_paper.pdf)]
412. **Large-scale long-tailed recognition in an open world.**<br>
*Liu, Ziwei and Miao, Zhongqi and Zhan, Xiaohang and Wang, Jiayun and Gong, Boqing and Yu, Stella X.*<br>
CVPR 2019. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_Large-Scale_Long-Tailed_Recognition_in_an_Open_World_CVPR_2019_paper.pdf)]
413. **Byzantine-Robust Learning on Heterogeneous Datasets via Bucketing.**<br>
*Karimireddy, Sai Praneeth and He, Lie and Jaggi, Martin.*<br>
ICLR 2022. [[Paper](https://openreview.net/pdf?id=jXKKDEi5vJt)]
414. **Fairness in graph mining: A survey.**<br>
*Dong, Yushun and Ma, Jing and Wang, Song and Chen, Chen and Li, Jundong.*<br>
TKDE 2023. [[Paper](https://ieeexplore.ieee.org/abstract/document/10097603)]
415. **Can we achieve robustness from data alone?**<br>
*Tsilivis, Nikolaos and Su, Jingtong and Kempe, Julia.*<br>
arXiv 2023. [[Paper](https://arxiv.org/abs/2207.11727)]
## Connections via similarity or opposition 
416. **MESA: boost ensemble imbalanced learning with meta-sampler.**<br>
*Liu, Zhining and Wei, Pengfei and Jiang, Jing and Cao, Wei and Bian, Jiang and Chang, Yi.*<br>
NeurIPS 2020. [[Paper](https://proceedings.neurips.cc/paper/2020/file/a64bd53139f71961c5c31a9af03d775e-Paper.pdf)]
417. **Background data resampling for outlier-aware classification.**<br>
*Li, Yi and Vasconcelos, Nuno.*<br>
CVPR 2020. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Background_Data_Resampling_for_Outlier-Aware_Classification_CVPR_2020_paper.pdf)]
## Connections via theory 
418. **Which Samples Should Be Learned First: Easy or Hard?**<br>
*Which Samples Should Be Learned First: Easy or Hard?*<br>
TNNLS 2023. [[Paper](https://ieeexplore.ieee.org/abstract/document/10155763)]
419. **PANDA: AdaPtive noisy data augmentation for regularization of undirected graphical models.**<br>
*Li, Yinan and Liu, Xiao and Liu, Fang.*<br>
arXiv 2019. [[Paper](https://arxiv.org/abs/1810.04851)]
420. **Understanding robust overfitting of adversarial training and beyond.**<br>
*Yu, Chaojian and Han, Bo and Shen, Li and Yu, Jun and Gong, Chen and Gong, Mingming and Liu, Tongliang.*<br>
ICML 2022. [[Paper](https://proceedings.mlr.press/v162/yu22b/yu22b.pdf)]
421. **Stability analysis and generalization bounds of adversarial training.**<br>
*Xiao, Jiancong and Fan, Yanbo and Sun, Ruoyu and Wang, Jue and Luo, Zhi-Quan.*<br>
NeurIPS 2022. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/637de5e2a7a77f741b0b84bd61c83125-Paper-Conference.pdf)]
422. **Understanding the role of importance weighting for deep learning.**<br>
*Xu, Da and Ye, Yuting and Ruan, Chuanwei.*<br>
arXiv 2021. [[Paper](https://arxiv.org/abs/2103.15209)]  
# Future Directions                                                                           
## Principles of data optimization
423. **Why resampling outperforms reweighting for correcting sampling bias with stochastic gradients.**<br>
*An, Jing and Ying, Lexing and Zhu, Yuhua.*<br>
ICLR 2021. [[Paper](https://openreview.net/pdf?id=iQQK02mxVIT)]
424. **The class imbalance problem.**<br>
*Megahed, Fadel M and Chen, Ying-Ju and Megahed, Aly and Ong, Yuya and Altman, Naomi and Krzywinski, Martin.*<br>
Nature Methods 2021. [[Paper](https://www.nature.com/articles/s41592-021-01302-4)]    
425. **Gaussian distribution based oversampling for imbalanced data classification.**<br>
*Xie, Yuxi and Qiu, Min and Zhang, Haibo and Peng, Lizhi and Chen, Zhenxiang.*<br>
TKDE 2022. [[Paper](https://ieeexplore.ieee.org/abstract/document/9063462)]  
426. **Which Samples Should Be Learned First: Easy or Hard?**<br>
*Which Samples Should Be Learned First: Easy or Hard?*<br>
TNNLS 2023. [[Paper](https://ieeexplore.ieee.org/abstract/document/10155763)]
427. **Generalized Entropy Regularization or: There’s Nothing Special about Label Smoothing.**<br>
*Meister, Clara and Salesky, Elizabeth and Cotterell, Ryan.*<br>
ACL 2020. [[Paper](https://aclanthology.org/2020.acl-main.615/)]
428. **Towards understanding label smoothing.**<br>
*Xu, Yi and Xu, Yuanhong and Qian, Qi and Li, Hao and Jin, Rong.*<br>
arXiv 2020. [[Paper](https://arxiv.org/abs/2006.11653)]
429. **When does label smoothing help?.**<br>
*Müller, Rafael and Kornblith, Simon and Hinton, Geoffrey E.*<br>
NeurIPS 2019. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/f1748d6b0fd9d439f71450117eba2725-Paper.pdf)]
430. **An investigation of how label smoothing affects generalization.**<br>
*Chen, Blair and Ziyin, Liu and Wang, Zihao and Liang, Paul Pu.*<br>
arXiv 2020. [[Paper](https://arxiv.org/abs/2010.12648)]
431. **Adversarial examples are not bugs, they are features.**<br>
*Ilyas, Andrew and Santurkar, Shibani and Tsipras, Dimitris and Engstrom, Logan and Tran, Brandon and Madry, Aleksander.*<br>
NeurIPS 2019. [[Paper](https://proceedings.neurips.cc/paper/2019/file/e2c420d928d4bf8ce0ff2ec19b371514-Paper.pdf)] 
432. **A survey of robust adversarial training in pattern recognition: Fundamental, theory, and methodologies.**<br>
*Qian, Zhuang and Huang, Kaizhu and Wang, Qiu-Feng and Zhang, Xu-Yao.*<br>
Pattern Recognition 2022. [[Paper](https://www.sciencedirect.com/science/article/pii/S0031320322003703?casa_token=LJNYiR_ye3wAAAAA:_RKKYApc4M26_UmHBB5JNNxe5pVuw5-zkEAC8Kh8q78AtlUeYNRkDA4_tKtNlvUYoBs6d_HW98Y)]
433. **ReduNet: A white-box deep network from the principle of maximizing rate reduction.**<br>
*Chan, Kwan Ho Ryan and Yu, Yaodong and You, Chong and Qi, Haozhi and Wright, John and Ma, Yi.*<br>
JMLR 2022. [[Paper](https://www.jmlr.org/papers/volume23/21-0631/21-0631.pdf)]  
434. **Combining Ensembles and Data Augmentation Can Harm Your Calibration.**<br>
*Wen, Yeming and Jerfel, Ghassen and Muller, Rafael and Dusenberry, Michael W and Snoek, Jasper and Lakshminarayanan, Balaji and Tran, Dustin.*<br>
ICLR 2021. [[Paper](https://openreview.net/pdf?id=g11CZSghXyY)]  
435. **Does label smoothing mitigate label noise?**<br>
*Lukasik, Michal and Bhojanapalli, Srinadh and Menon, Aditya and Kumar, Sanjiv.*<br>
ICML 2020. [[Paper](https://proceedings.mlr.press/v119/lukasik20a/lukasik20a.pdf)]                                   
## Interpretable data optimization
436. **Techniques for interpretable machine learning.**<br>
*Du, Mengnan and Liu, Ninghao and Hu, Xia.*<br>
Communications of the ACM 2019. [[Paper](https://dl.acm.org/doi/fullHtml/10.1145/3359786)]  
437. **Can Perceptual Guidance Lead to Semantically Explainable Adversarial Perturbations?**<br>
*Pochimireddy, Charantej Reddy and Siripuram, Aditya T and Channappayya, Sumohana S.*<br>
IEEE Journal of Selected Topics in Signal Processing 2023. [[Paper](https://ieeexplore.ieee.org/abstract/document/10073613)]  
438. **Towards explaining the effects of data preprocessing on machine learning.**<br>
*Zelaya, Carlos Vladimiro González.*<br>
ICDE 2019. [[Paper](https://ieeexplore.ieee.org/abstract/document/8731532)]  
## Human-in-the-loop data optimization
439. **Human-in-the-loop machine learning: A state of the art.**<br>
*Mosqueira-Rey, Eduardo and Hernández-Pereira, Elena and Alonso-Ríos, David and Bobes-Bascarán, José and Fernández-Leal, Ángel.*<br>
Artificial Intelligence Review 2023. [[Paper](https://link.springer.com/article/10.1007/s10462-022-10246-w)]  
440. **Human-in-the-loop mixup.**<br>
*Collins, Katherine M and Bhatt, Umang and Liu, Weiyang and Piratla, Vihari and Sucholutsky, Ilia and Love, Bradley and Weller, Adrian.*<br>
UAI 2023. [[Paper](https://proceedings.mlr.press/v216/collins23a/collins23a.pdf)]  
441. **Trick me if you can: Human-in-the-loop generation of adversarial examples for question answering.**<br>
*Wallace, Eric and Rodriguez, Pedro and Feng, Shi and Yamada, Ikuya and Boyd-Graber, Jordan.*<br>
TACL 2019. [[Paper](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00279/43493/Trick-Me-If-You-Can-Human-in-the-Loop-Generation)]  
442. **Estimating example difficulty using variance of gradients.**<br>
*Agarwal, Chirag and D'souza, Daniel and Hooker, Sara.*<br>
CVPR 2022. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Agarwal_Estimating_Example_Difficulty_Using_Variance_of_Gradients_CVPR_2022_paper.pdf)]  
## Data optimization for new challenges   
443. **Ngc: A unified framework for learning with open-world noisy data.**<br>
*Wu, Zhi-Fan and Wei, Tong and Jiang, Jianwen and Mao, Chaojie and Tang, Mingqian and Li, Yu-Feng.*<br>
ICCV 2021. [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_NGC_A_Unified_Framework_for_Learning_With_Open-World_Noisy_Data_ICCV_2021_paper.pdf)] 
444. **Improving contrastive learning on imbalanced data via open-world sampling.**<br>
*Jiang, Ziyu and Chen, Tianlong and Chen, Ting and Wang, Zhangyang.*<br>
NeurIPS 2021. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2021/file/2f37d10131f2a483a8dd005b3d14b0d9-Paper.pdf)]          
445. **Generalized but not Robust? Comparing the Effects of Data Modification Methods on Out-of-Domain Generalization and Adversarial Robustness.**<br>
*Gokhale, Tejas and Mishra, Swaroop and Luo, Man and Sachdeva, Bhavdeep Singh and Baral, Chitta.*<br>
ACL Findings 2022. [[Paper](https://asu.elsevierpure.com/en/publications/generalized-but-not-robust-comparing-the-effects-of-data-modifica)] 
446. **A survey of large language models.**<br>
*Zhao, Wayne Xin and Zhou, Kun and Li, Junyi and Tang, Tianyi and Wang, Xiaolei and Hou, Yupeng and Min, Yingqian and Zhang, Beichen and Zhang, Junjie and Dong, Zican and  Du, Yifan and Yang, Chen and Chen, Yushuo and Chen, Zhipeng and Jiang, Jinhao and Ren, Ruiyang and Li, Yifan and Tang, Xinyu and Liu, Zikang and Liu, Peiyu and Nie, Jian-Yun and Wen, Ji-Rong.*<br>
arXiv 2023. [[Paper](https://arxiv.org/abs/2303.18223)] 
447. **Instructiongpt-4: A 200-instruction paradigm for fine-tuning minigpt-4.**<br>
*Wei, Lai and Jiang, Zihao and Huang, Weiran and Sun, Lichao.*<br>
arXiv 2023. [[Paper](https://arxiv.org/abs/2308.12067)] 
448. **Calibrating Language Models via Augmented Prompt Ensembles.**<br>
*Jiang, Mingjian and Ruan, Yangjun and Huang, Sicong and Liao, Saifei and Pitis, Silviu and Grosse, Roger Baker and Ba, Jimmy.*<br>
ICML Workshop 2023. [[Paper](https://openreview.net/pdf?id=L0dc4wqbNs)] 
449. **Multimodal machine learning: A survey and taxonomy.**<br>
*Baltrušaitis, Tadas and Ahuja, Chaitanya and Morency, Louis-Philippe.*<br>
TPAMI 2019. [[Paper](https://ieeexplore.ieee.org/abstract/document/8269806)] 
450. **Learning Robust Multi-Modal Representation for Multi-Label Emotion Recognition via Adversarial Masking and Perturbation.**<br>
*Ge, Shiping and Jiang, Zhiwei and Cheng, Zifeng and Wang, Cong and Yin, Yafeng and Gu, Qing.*<br>
WWW 2023. [[Paper](https://cs.nju.edu.cn/_upload/tpl/02/e0/736/template736/YinPaper/WWW2023Ge.pdf)] 
## Data optimization agent 
451. **Automatic data augmentation via invariance-constrained learning.**<br>
*Hounie, Ignacio and Chamon, Luiz FO and Ribeiro, Alejandro.*<br>
ICML 2023. [[Paper](https://proceedings.mlr.press/v202/hounie23a/hounie23a.pdf)] 
452. **Importantaug: a data augmentation agent for speech.**<br>
*Trinh, Viet Anh and Kavaki, Hassan Salami and Mandel, Michael I.*<br>
ICASSP 2022. [[Paper](https://ieeexplore.ieee.org/abstract/document/9747003)]  
453. **Autobalance: Optimized loss functions for imbalanced data.**<br>
*Li, Mingchen and Zhang, Xuechen and Thrampoulidis, Christos and Chen, Jiasi and Oymak, Samet.*<br>
NeurIPS 2021. [[Paper](https://proceedings.neurips.cc/paper/2021/file/191f8f858acda435ae0daf994e2a72c2-Paper.pdf)]
