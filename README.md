# Data Optimization in Deep Learning: A Survey
:hugs: The repository is a collection of resources on data optimization in deep learning, serving as a supplement to our survey paper "Data Optimization in Deep Learning: A Survey". If you have any recommendations for missing work and any suggestions, please feel free to [pull requests](https://github.com/YaoRujing/Data-Optimization/pulls) or [contact us](mailto:wuou@tju.edu.cn):envelope:.

If you find our survey useful, please kindly cite our paper:

```bibtex
@article{wu2023data,
      title={Data Optimization in Deep Learning: A Survey}, 
      author={Wu, Ou and Yao, Rujing},
      year={2023}
}
```

**📝 Table of Contents**
- [Introduction](#introduction)
- [Related studies](#related-studies)
- [The proposed taxonomy](#the-proposed-taxonomy)
- [Goals, scenarios, and data objects](#goals-scenarios-and-data-objects)
  - [Optimization goals](#optimization-goals)
  - [Application scenarios](#application-scenarios)
  - [Data objects](#data-objects)
    - [Primary objects](#primary-objects)
    - [Other objects](#other-objects)                                                             
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
    - [Assignment manner](#assignment-manner)                                                 
  - [Data pruning](#data-pruning)
    - [Dataset distillation](#dataset-distillation) 
    - [Subset selection](#subset-selection)                                                                   
  - [Other typical techniques](#other-typical-techniques)   
    - [Pure mathematical optimization](#mathematical-optimization) 
    - [Technique combination](#technique-combination)                                                                          
- [Data optimization theories](#data-optimization-theories)
  - [Formalization](#formalization)                                          
  - [Explanation](#explanation)
- [Connections among different techniques](#connections-among-different-techniques)
  - [Connections via data perception](#Connection-via-data-perception)                                          
  - [Connections via application scenarios](#connections-via-application-scenarios)
  - [Connections via similarity/opposition](#connections-via-similarity-opposition)
  - [Connections via theory](#connections-via-theory)
- [Future Directions](#future-directions)
  - [Principles of data optimization](#principles-of-data-optimization)                                          
  - [Interpretable data optimization](#interpretable-data-optimization)
  - [Human-in-the-loop data optimization](#human-in-the-loop-data-optimization)
  - [Data optimization for new challenges](#data-optimization-for-new-challenges)            
  - [Data optimization agent](#data-optimization-agent) 
                                              

# Introduction
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
7. **Can Data Diversity Enhance Learning Generalization?.**<br>
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
*Algan, G{\"o}rkem and Ulusoy, Ilkay.*<br>
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
21. **Learning under concept drift: A review.**<br>
*Lu, Jie and Liu, Anjin and Dong, Fan and Gu, Feng and Gama, Joao and Zhang, Guangquan.*<br>
TKDE 2018. [[Paper](https://ieeexplore.ieee.org/abstract/document/8496795)]
22. **Recent Advances in Concept Drift Adaptation Methods for Deep Learning.**<br>
*Yuan, Liheng and Li, Heng and Xia, Beihao and Gao, Cuiying and Liu, Mingyue and Yuan, Wei and You, Xinge.*<br>
IJCAI 2022. [[Paper](https://www.ijcai.org/proceedings/2022/0788.pdf)]
23. **Adaptive dendritic cell-deep learning approach for industrial prognosis under changing conditions.**<br>
*Diez-Olivan, Alberto and Ortego, Patxi and Del Ser, Javier and Landa-Torres, Itziar and Galar, Diego and Camacho, David and Sierra, Basilio.*<br>
IEEE Transactions on Industrial Informatics 2021. [[Paper](https://ieeexplore.ieee.org/abstract/document/9352529)]
24. **A survey on concept drift adaptation.**<br>
*Gama, Jo{\~a}o and {\v{Z}}liobait{\.e}, Indr{\.e} and Bifet, Albert and Pechenizkiy, Mykola and Bouchachia, Abdelhamid.*<br>
ACM computing surveys 2014. [[Paper](https://dl.acm.org/doi/abs/10.1145/2523813)]
25. **An overview on concept drift learning.**<br>
*Iwashita, Adriana Sayuri and Papa, Joao Paulo.*<br>
IEEE access 2018. [[Paper](https://ieeexplore.ieee.org/abstract/document/8571222)]
26. **Opportunities and challenges in deep learning adversarial robustness: A survey.**<br>
*Silva, Samuel Henrique and Najafirad, Peyman.*<br>
arXiv 2020. [[Paper](https://arxiv.org/abs/2007.00753)]
27. **Robustness of deep learning models on graphs: A survey.**<br>
*Xu, Jiarong and Chen, Junru and You, Siqi and Xiao, Zhiqing and Yang, Yang and Lu, Jiangang.*<br>
AI Open 2021. [[Paper](https://www.sciencedirect.com/science/article/pii/S2666651021000139)]
28. **A survey of adversarial defenses and robustness in nlp.**<br>
*Goyal, Shreya and Doddapaneni, Sumanth and Khapra, Mitesh M and Ravindran, Balaraman.*<br>
ACM Computing Surveys 2023. [[Paper](https://dl.acm.org/doi/abs/10.1145/3593042)]
29. **A survey on bias and fairness in machine learning.**<br>
*Mehrabi, Ninareh and Morstatter, Fred and Saxena, Nripsuta and Lerman, Kristina and Galstyan, Aram.*<br>
ACM computing surveys 2021. [[Paper](https://dl.acm.org/doi/abs/10.1145/3457607)]
30. **FAIR: Fair adversarial instance re-weighting.**<br>
*Petrovi{\'c}, Andrija and Nikoli{\'c}, Mladen and Radovanovi{\'c}, Sandro and Deliba{\v{s}}i{\'c}, Boris and Jovanovi{\'c}, Milo{\v{s}}.*<br>
Neurocomputing 2022. [[Paper](https://www.sciencedirect.com/science/article/pii/S0925231221019408?casa_token=zl0smR7i06AAAAAA:ybSefSP57QrNHVMLB9lb4rTQLCubIPA2Ggnh87bSC3Dv4faAC4f2zg5a38HQwA-6OyDUVpIK4C4)]
31. **Trustworthiness of autonomous systems.**<br>
*Devitt, S.*<br>
Foundations of trusted autonomy 2018. [[Paper](https://link.springer.com/chapter/10.1007/978-3-319-64816-3_9)]
32. **Trustworthy artificial intelligence: a review.**<br>
*Kaur, Davinder and Uslu, Suleyman and Rittichier, Kaley J and Durresi, Arjan.*<br>
ACM Computing Surveys 2022. [[Paper](https://dl.acm.org/doi/abs/10.1145/3491209)]
33. **Trustworthy Graph Learning: Reliability, Explainability, and Privacy Protection.**<br>
*Wu, Bingzhe and Bian, Yatao and Zhang, Hengtong and Li, Jintang and Yu, Junchi and Chen, Liang and Chen, Chaochao and Huang, Junzhou.*<br>
SIGKDD 2022. [[Paper](https://dl.acm.org/doi/abs/10.1145/3534678.3542597)]
34. **Combating Noisy Labels in Long-Tailed Image Classification.**<br>
*Fang, Chaowei and Cheng, Lechao and Qi, Huiyan and Zhang, Dingwen.*<br>
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.00273)]
35. **An Empirical Study of Accuracy, Fairness, Explainability, Distributional Robustness, and Adversarial Robustness.**<br>
*Singh, Moninder and Ghalachyan, Gevorg and Varshney, Kush R and Bryant, Reginald E.*<br>
arXiv 2021. [[Paper](https://arxiv.org/abs/2109.14653)]
36. **A Survey of Data Optimization for Problems in Computer Vision Datasets.**<br>
*Wan, Zhijing and Wang, Zhixiang and Chung, CheukTing and Wang, Zheng.*<br>
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.11717)]
37. **Towards Data-centric Graph Machine Learning: Review and Outlook.**<br>
*Zheng, Xin and Liu, Yixin and Bao, Zhifeng and Fang, Meng and Hu, Xia and Liew, Alan Wee-Chung and Pan, Shirui.*<br>
arXiv 2023. [[Paper](https://arxiv.org/abs/2309.10979)]
# The proposed taxonomy
# Goals, scenarios, and data objects
## Optimization goals
38. **G-softmax: improving intraclass compactness and interclass separability of features.**<br>
*Luo, Yan and Wong, Yongkang and Kankanhalli, Mohan and Zhao, Qi.*<br>
TNNLS 2019. [[Paper](https://ieeexplore.ieee.org/ielaam/5962385/8984609/8712413-aam.pdf)]
39. **Label noise sgd provably prefers flat global minimizers.**<br>
*Damian, Alex and Ma, Tengyu and Lee, Jason D.*<br>
NeurIPS 2021. [[Paper](https://proceedings.neurips.cc/paper/2021/file/e6af401c28c1790eaef7d55c92ab6ab6-Paper.pdf)]
40. **Implicit semantic data augmentation for deep networks.**<br>
*Wang, Yulin and Pan, Xuran and Song, Shiji and Zhang, Hong and Huang, Gao and Wu, Cheng.*<br>
NeurIPS 2019. [[Paper](https://proceedings.neurips.cc/paper/2019/file/15f99f2165aa8c86c9dface16fefd281-Paper.pdf)]
41. **Meta balanced network for fair face recognition.**<br>
*Wang, Mei and Zhang, Yaobin and Deng, Weihong.*<br>
TPAMI 2021. [[Paper](https://ieeexplore.ieee.org/abstract/document/9512390)]
42. **Data Augmentation by Selecting Mixed Classes Considering Distance Between Classes.**<br>
*Fujii, Shungo and Ishii, Yasunori and Kozuka, Kazuki and Hirakawa, Tsubasa and Yamashita, Takayoshi and Fujiyoshi, Hironobu.*<br>
arXiv 2022. [[Paper](https://arxiv.org/abs/2209.05122)]
43. **mixup: Beyond Empirical Risk Minimization.**<br>
*Zhang, Hongyi and Cisse, Moustapha and Dauphin, Yann N and Lopez-Paz, David.*<br>
ICLR 2018. [[Paper](https://openreview.net/pdf?id=r1Ddp1-Rb)]
44. **Online batch selection for faster training of neural networks.**<br>
*Loshchilov, Ilya and Hutter, Frank.*<br>
ICLR workshop track 2016. [[Paper](https://openreview.net/forum?id=r8lrkABJ7H8wknpYt5KB)]
45. **Fair Mixup: Fairness via Interpolation.**<br>
*Mroueh, Youssef and others.*<br>
ICLR 2021. [[Paper](https://openreview.net/pdf?id=DNl5s5BXeBn)]
46. **Fair classification with adversarial perturbations.**<br>
*Celis, L Elisa and Mehrotra, Anay and Vishnoi, Nisheeth.*<br>
NeurIPS 2021. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2021/file/44e207aecc63505eb828d442de03f2e9-Paper.pdf)]
47. **FORML: Learning to Reweight Data for Fairness.**<br>
*Yan, Bobby and Seto, Skyler and Apostoloff, Nicholas.*<br>
arXiv 2022. [[Paper](https://arxiv.org/abs/2202.01719)]
48. **Class-balanced loss based on effective number of samples.**<br>
*Cui, Yin and Jia, Menglin and Lin, Tsung-Yi and Song, Yang and Belongie, Serge.*<br>
CVPR 2019. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf)]
49. **Logit perturbation.**<br>
*Li, Mengyang and Su, Fengguang and Wu, Ou and Zhang, Ji.*<br>
AAAI 2022. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/20024)]
50. **Scale-aware automatic augmentations for object detection with dynamic training.**<br>
*Chen, Yukang and Zhang, Peizhen and Kong, Tao and Li, Yanwei and Zhang, Xiangyu and Qi, Lu and Sun, Jian and Jia, Jiaya.*<br>
TPAMI 2023. [[Paper](https://ieeexplore.ieee.org/abstract/document/9756374)]
51. **Obtaining well calibrated probabilities using bayesian binning.**<br>
*Naeini, Mahdi Pakdaman and Cooper, Gregory and Hauskrecht, Milos.*<br>
AAAI 2015. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/9602)]
52. **The devil is in the margin: Margin-based label smoothing for network calibration.**<br>
*Liu, Bingyuan and Ben Ayed, Ismail and Galdran, Adrian and Dolz, Jose.*<br>
CVPR 2022. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_The_Devil_Is_in_the_Margin_Margin-Based_Label_Smoothing_for_CVPR_2022_paper.pdf)]
53. **Calibrating deep neural networks using focal loss.**<br>
*Mukhoti, Jishnu and Kulharia, Viveka and Sanyal, Amartya and Golodetz, Stuart and Torr, Philip and Dokania, Puneet.*<br>
NeurIPS 2020. [[Paper](https://proceedings.neurips.cc/paper/2020/file/aeb7b30ef1d024a76f21a1d40e30c302-Paper.pdf)]
## Application scenarios
54. **Can Data Diversity Enhance Learning Generalization?**<br>
*Yu, Yu and Khadivi, Shahram and Xu, Jia.*<br>
COLING 2022. [[Paper](https://aclanthology.org/2022.coling-1.437.pdf)]
55. **Diversify your vision datasets with automatic diffusion-based augmentation.**<br>
*Dunlap, Lisa and Umino, Alyssa and Zhang, Han and Yang, Jiezhi and Gonzalez, Joseph E and Darrell, Trevor.*<br>
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16289)]
56. **Infusing definiteness into randomness: rethinking composition styles for deep image matting.**<br>
*Ye, Zixuan and Dai, Yutong and Hong, Chaoyi and Cao, Zhiguo and Lu, Hao.*<br>
AAAI 2023. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/25432)]
57. **Image data augmentation for deep learning: A survey.**<br>
*Yang, Suorong and Xiao, Weikang and Zhang, Mengcheng and Guo, Suhan and Zhao, Jian and Shen, Furao.*<br>
arXiv 2022. [[Paper](https://arxiv.org/abs/2204.08610)]
58. **BigTranslate: Augmenting Large Language Models with Multilingual Translation Capability over 100 Languages.**<br>
*Yang, Wen and Li, Chong and Zhang, Jiajun and Zong, Chengqing.*<br>
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.18098)]
59. **Adversarial training for large neural language models.**<br>
*Liu, Xiaodong and Cheng, Hao and He, Pengcheng and Chen, Weizhu and Wang, Yu and Poon, Hoifung and Gao, Jianfeng.*<br>
arXiv 2020. [[Paper](https://arxiv.org/abs/2004.08994)]
## Data objects
### Primary objects
### Other objects
60. **Understanding the difficulty of training deep feedforward neural networks.**<br>
*Glorot, Xavier and Bengio, Yoshua.*<br>
AISTATS 2010. [[Paper](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)]
61. **Delving deep into rectifiers: Surpassing human-level performance on imagenet classification.**<br>
*He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian.*<br>
ICCV 2015. [[Paper](https://openaccess.thecvf.com/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf)]
62. **Submodular Meta Data Compiling for Meta Optimization.**<br>
*Su, Fengguang and Zhu, Yu and Wu, Ou and Deng, Yingjun.*<br>
ECML/PKDD 2022. [[Paper](https://2022.ecmlpkdd.org/wp-content/uploads/2022/09/sub_474.pdf)]
# Optimization pipeline
## Data perception 
### Perception on different granularity levels
63. **O2u-net: A simple noisy label detection approach for deep neural networks.**<br>
*Huang, Jinchi and Qu, Lie and Jia, Rongfei and Zhao, Binqiang.*<br>
ICCV 2019. [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_O2U-Net_A_Simple_Noisy_Label_Detection_Approach_for_Deep_Neural_ICCV_2019_paper.pdf)] 
64. **Gradient harmonized single-stage detector.**<br>
*Li, Buyu and Liu, Yu and Wang, Xiaogang.*<br>
AAAI 2019. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/4877)]
65. **Delving deep into label smoothing.**<br>
*Zhang, Chang-Bin and Jiang, Peng-Tao and Hou, Qibin and Wei, Yunchao and Han, Qi and Li, Zhen and Cheng, Ming-Ming.*<br>
TIP 2021. [[Paper](https://ieeexplore.ieee.org/abstract/document/9464693)]
66. **Class-wise difficulty-balanced loss for solving class-imbalance.**<br>
*Sinha, Saptarshi and Ohashi, Hiroki and Nakamura, Katsuyuki.*<br>
ACCV 2020. [[Paper](https://openaccess.thecvf.com/content/ACCV2020/papers/Sinha_Class-Wise_Difficulty-Balanced_Loss_for_Solving_Class-Imbalance_ACCV_2020_paper.pdf)]
67. **Ccl: Class-wise curriculum learning for class imbalance problems.**<br>
*Escudero-Vi{\~n}olo, Marcos and L{\'o}pez-Cifuentes, Alejandro.*<br>
ICIP 2022. [[Paper](https://ieeexplore.ieee.org/abstract/document/9897273)]
68. **Hyper-sausage coverage function neuron model and learning algorithm for image classification.**<br>
*Ning, Xin and Tian, Weijuan and He, Feng and Bai, Xiao and Sun, Le and Li, Weijun.*<br>
Pattern Recognition 2023. [[Paper](https://www.sciencedirect.com/science/article/pii/S0031320322006951)]
69. **Measuring the effect of training data on deep learning predictions via randomized experiments.**<br>
*Lin, Jinkun and Zhang, Anqi and L{\'e}cuyer, Mathias and Li, Jinyang and Panda, Aurojit and Sen, Siddhartha.*<br>
ICML 2022. [[Paper](https://proceedings.mlr.press/v162/lin22h/lin22h.pdf)]
70. **Combining Adversaries with Anti-adversaries in Training.**<br>
*Zhou, Xiaoling and Yang, Nan and Wu, Ou.*<br>
AAAI 2023. [[Paper](https://arxiv.org/abs/2304.12550)]
### Perception on different types                        
71. **Implicit semantic data augmentation for deep networks.**<br>
*Wang, Yulin and Pan, Xuran and Song, Shiji and Zhang, Hong and Huang, Gao and Wu, Cheng.*<br>
NeurIPS 2019. [[Paper](https://proceedings.neurips.cc/paper/2019/file/15f99f2165aa8c86c9dface16fefd281-Paper.pdf)]
72. **Invariant feature learning for generalized long-tailed classification.**<br>
*Tang, Kaihua and Tao, Mingyuan and Qi, Jiaxin and Liu, Zhenguang and Zhang, Hanwang.*<br>
ECCV 2022. [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-20053-3_41)]
73. **DatasetEquity: Are All Samples Created Equal? In The Quest For Equity Within Datasets.**<br>
*Shrivastava, Shubham and Zhang, Xianling and Nagesh, Sushruth and Parchami, Armin.*<br>
ICCV 2023. [[Paper](https://openaccess.thecvf.com/content/ICCV2023W/OODCV/papers/Shrivastava_DatasetEquity_Are_All_Samples_Created_Equal_In_The_Quest_For_ICCVW_2023_paper.pdf)]
74. **Tackling the imbalance for gnns.**<br>
*Wang, Rui and Xiong, Weixuan and Hou, Qinghu and Wu, Ou.*<br>
IJCNN 2022. [[Paper](https://ieeexplore.ieee.org/abstract/document/9892713)]
75. **O2u-net: A simple noisy label detection approach for deep neural networks.**<br>
*Huang, Jinchi and Qu, Lie and Jia, Rongfei and Zhao, Binqiang.*<br>
ICCV 2019. [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_O2U-Net_A_Simple_Noisy_Label_Detection_Approach_for_Deep_Neural_ICCV_2019_paper.pdf)]
76. **Curriculum learning.**<br>
*Bengio, Yoshua and Louradour, J{\'e}r{\^o}me and Collobert, Ronan and Weston, Jason.*<br>
ICML 2009. [[Paper](https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/15972/Bengio%2C%202009%20Curriculum%20Learning.pdf?sequence=1&isAllowed=y)]
77. **Focal loss for dense object detection.**<br>
*Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr.*<br>
ICCV 2017. [[Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)]
78. **Deep learning on a data diet: Finding important examples early in training.**<br>
*Paul, Mansheej and Ganguli, Surya and Dziugaite, Gintare Karolina.*<br>
NeurIPS 2021. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2021/file/ac56f8fe9eea3e4a365f29f0f1957c55-Paper.pdf)]
79. **Exploring the Learning Difficulty of Data Theory and Measure.**<br>
*Zhu, Weiyao and Wu, Ou and Su, Fengguang and Deng, Yingjun.*<br>
arXiv 2022. [[Paper](https://arxiv.org/abs/2205.07427)]
80. **Beyond neural scaling laws: beating power law scaling via data pruning.**<br>
*Sorscher, Ben and Geirhos, Robert and Shekhar, Shashank and Ganguli, Surya and Morcos, Ari.*<br>
NeurIPS 2022. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/7b75da9b61eda40fa35453ee5d077df6-Paper-Conference.pdf)]
81. **A review of uncertainty quantification in deep learning: Techniques, applications and challenges.**<br>
*Abdar, Moloud and Pourpanah, Farhad and Hussain, Sadiq and Rezazadegan, Dana and Liu, Li and Ghavamzadeh, Mohammad and Fieguth, Paul and Cao, Xiaochun and Khosravi, Abbas and Acharya, U Rajendra and Makarenkov, Vladimir and Nahavandi, Saeid.*<br>
Information fusion 2021. [[Paper](https://www.sciencedirect.com/science/article/pii/S1566253521001081)]
82. **A tale of two long tails.**<br>
*D'souza, Daniel and Nussbaum, Zach and Agarwal, Chirag and Hooker, Sara.*<br>
arXiv 2021. [[Paper](https://arxiv.org/abs/2107.13098)]
83. **What uncertainties do we need in bayesian deep learning for computer vision?**<br>
*Kendall, Alex and Gal, Yarin.*<br>
NeurIPS 2017. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/2650d6089a6d640c5e85b2b88265dc2b-Paper.pdf)]
84. **Submodular optimization-based diverse paraphrasing and its effectiveness in data augmentation.**<br>
*Kumar, Ashutosh and Bhattamishra, Satwik and Bhandari, Manik and Talukdar, Partha.*<br>
NAACL 2019. [[Paper](https://aclanthology.org/N19-1363/)]
85. **Submodular Meta Data Compiling for Meta Optimization.**<br>
*Su, Fengguang and Zhu, Yu and Wu, Ou and Deng, Yingjun.*<br>
ECML/PKDD 2022. [[Paper](https://2022.ecmlpkdd.org/wp-content/uploads/2022/09/sub_474.pdf)]
86. **The vendi score: A diversity evaluation metric for machine learning.**<br>
*Friedman, Dan and Dieng, Adji Bousso.*<br>
arXiv 2023. [[Paper](https://arxiv.org/abs/2210.02410)]
87. **Improved techniques for training gans.**<br>
*Salimans, Tim and Goodfellow, Ian and Zaremba, Wojciech and Cheung, Vicki and Radford, Alec and Chen, Xi.*<br>
NeurIPS 2016. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2016/file/8a3363abe792db2d8761d6403605aeb7-Paper.pdf)]
88. **Rethinking Class Imbalance in Machine Learning.**<br>
*Wu, Ou.*<br>
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.03900)]  
89. **Invariant feature learning for generalized long-tailed classification.**<br>
*Tang, Kaihua and Tao, Mingyuan and Qi, Jiaxin and Liu, Zhenguang and Zhang, Hanwang.*<br>
ECCV 2022. [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-20053-3_41)]
90. **Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics.**<br>
*Swayamdipta, Swabha and Schwartz, Roy and Lourie, Nicholas and Wang, Yizhong and Hajishirzi, Hannaneh and Smith, Noah A and Choi, Yejin.*<br>
EMNLP 2020. [[Paper](https://aclanthology.org/2020.emnlp-main.746/)]
91. **Learning with neighbor consistency for noisy labels.**<br>
*Iscen, Ahmet and Valmadre, Jack and Arnab, Anurag and Schmid, Cordelia.*<br>
CVPR 2022. [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Iscen_Learning_With_Neighbor_Consistency_for_Noisy_Labels_CVPR_2022_paper.pdf)]
92. **An Empirical Study of Example Forgetting during Deep Neural Network Learning.**<br>
*Toneva, Mariya and Sordoni, Alessandro and des Combes, Remi Tachet and Trischler, Adam and Bengio, Yoshua and Gordon, Geoffrey J.*<br>
ICLR 2019. [[Paper](https://openreview.net/forum?id=BJlxm30cKm&fbclid=IwAR3kUvKWW-NyzCi7dB_zL47J_KJSMtcqQp8eFQEd5R07VWj5dcCwHJsXRcc)]
93. **Attaining Class-Level Forgetting in Pretrained Model Using Few Samples.**<br>
*Singh, Pravendra and Mazumder, Pratik and Karim, Mohammed Asad.*<br>
ECCV 2022. [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-19778-9_25)]
94. **Characterizing datapoints via second-split forgetting.**<br>
*Maini, Pratyush and Garg, Saurabh and Lipton, Zachary and Kolter, J Zico.*<br>
NeurIPS 2022. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/c20447998d6c624b4b97d4466a3bfff5-Paper-Conference.pdf)]  
95. **A comprehensive survey of forgetting in deep learning beyond continual learning.**<br>
*Wang, Zhenyi and Yang, Enneng and Shen, Li and Huang, Heng.*<br>
arXiv 2023. [[Paper](https://arxiv.org/abs/2307.09218)]  
96. **FINE samples for learning with noisy labels.**<br>
*Kim, Taehyeon and Ko, Jongwoo and Cho, Sangwook and Choi, Jinhwan and Yun, Se-Young.*<br>
NeurIPS 2021. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2021/file/ca91c5464e73d3066825362c3093a45f-Paper.pdf)]  
97. **A value for n-person games.**<br>
*L. S. Shapley.*<br>
Contributions to the Theory of Games 1953. [[Paper](https://apps.dtic.mil/sti/tr/pdf/AD0604084.pdf)]
98. **Data shapley: Equitable valuation of data for machine learning.**<br>
*Ghorbani, Amirata and Zou, James.*<br>
ICML 2019. [[Paper](https://proceedings.mlr.press/v97/ghorbani19c/ghorbani19c.pdf)]     
99. **Data valuation using reinforcement learning.**<br>
*Yoon, Jinsung and Arik, Sercan and Pfister, Tomas.*<br>
ICML 2020. [[Paper](https://proceedings.mlr.press/v119/yoon20a/yoon20a.pdf)] 
100. **Measuring the effect of training data on deep learning predictions via randomized experiments.**<br>
*Lin, Jinkun and Zhang, Anqi and L{\'e}cuyer, Mathias and Li, Jinyang and Panda, Aurojit and Sen, Siddhartha.*<br>
ICML 2022. [[Paper](https://proceedings.mlr.press/v162/lin22h/lin22h.pdf)]   
101. **Energy-Based Learning for Cooperative Games, with Applications to Valuation Problems in Machine Learning.**<br>
*Bian, Yatao and Rong, Yu and Xu, Tingyang and Wu, Jiaxiang and Krause, Andreas and Huang, Junzhou.*<br>
ICLR 2021. [[Paper](https://openreview.net/forum?id=xLfAgCroImw)]  
102. **OpenDataVal: a Unified Benchmark for Data Valuation.**<br>
*Jiang, Kevin Fu and Liang, Weixin and Zou, James and Kwon, Yongchan.*<br>
NeurIPS 2023. [[Paper](https://arxiv.org/abs/2306.10577)] 
103. **Locally adaptive label smoothing improves predictive churn.**<br>
*Bahri, Dara and Jiang, Heinrich.*<br>
ICML 2021. [[Paper](https://proceedings.mlr.press/v139/bahri21a/bahri21a.pdf)] 
104. **Data Profiling for Adversarial Training: On the Ruin of Problematic Data.**<br>
*Dong, Chengyu and Liu, Liyuan and Shang, Jingbo.*<br>
arXiv 2021. [[Paper](https://arxiv.org/abs/2102.07437v1)] 
105. **Training data influence analysis and estimation: A survey.**<br>
*Hammoudeh, Zayd and Lowd, Daniel.*<br>
arXiv 2023. [[Paper](https://arxiv.org/abs/2212.04612)]  
106. **Learning to purify noisy labels via meta soft label corrector.**<br>
*Wu, Yichen and Shu, Jun and Xie, Qi and Zhao, Qian and Meng, Deyu.*<br>
AAAI 2021. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/17244)] 
107. **Meta-weight-net: Learning an explicit mapping for sample weighting.**<br>
*Shu, Jun and Xie, Qi and Yi, Lixuan and Zhao, Qian and Zhou, Sanping and Xu, Zongben and Meng, Deyu.*<br>
NeurIPS 2019. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/e58cc5ca94270acaceed13bc82dfedf7-Paper.pdf)]
108. **Combining Adversaries with Anti-adversaries in Training.**<br>
*Zhou, Xiaoling and Yang, Nan and Wu, Ou.*<br>
AAAI 2023. [[Paper](https://arxiv.org/abs/2304.12550)]
### Static and dynamic perception    
109. **O2u-net: A simple noisy label detection approach for deep neural networks.**<br>
*Huang, Jinchi and Qu, Lie and Jia, Rongfei and Zhao, Binqiang.*<br>
ICCV 2019. [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_O2U-Net_A_Simple_Noisy_Label_Detection_Approach_for_Deep_Neural_ICCV_2019_paper.pdf)]
110. **Self-paced learning for latent variable models.**<br>
*Kumar, M and Packer, Benjamin and Koller, Daphne.*<br>
NeurIPS 2010. [[Paper](https://proceedings.neurips.cc/paper/2010/file/e57c6b956a6521b28495f2886ca0977a-Paper.pdf)]                    
## Analysis on perceived quantities
111. **An Empirical Study of Example Forgetting during Deep Neural Network Learning.**<br>
*Toneva, Mariya and Sordoni, Alessandro and des Combes, Remi Tachet and Trischler, Adam and Bengio, Yoshua and Gordon, Geoffrey J.*<br>
ICLR 2019. [[Paper](https://openreview.net/forum?id=BJlxm30cKm&fbclid=IwAR3kUvKWW-NyzCi7dB_zL47J_KJSMtcqQp8eFQEd5R07VWj5dcCwHJsXRcc)]     
112. **O2u-net: A simple noisy label detection approach for deep neural networks.**<br>
*Huang, Jinchi and Qu, Lie and Jia, Rongfei and Zhao, Binqiang.*<br>
ICCV 2019. [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_O2U-Net_A_Simple_Noisy_Label_Detection_Approach_for_Deep_Neural_ICCV_2019_paper.pdf)]
113. **Exploring the Learning Difficulty of Data Theory and Measure.**<br>
*Zhu, Weiyao and Wu, Ou and Su, Fengguang and Deng, Yingjun.*<br>
arXiv 2022. [[Paper](https://arxiv.org/abs/2205.07427)]
114. **Unsupervised label noise modeling and loss correction.**<br>
*Arazo, Eric and Ortego, Diego and Albert, Paul and O’Connor, Noel and McGuinness, Kevin.*<br>
ICML 2019. [[Paper](https://proceedings.mlr.press/v97/arazo19a/arazo19a.pdf)]  
115. **MILD: Modeling the Instance Learning Dynamics for Learning with Noisy Labels.**<br>
*Hu, Chuanyang and Yan, Shipeng and Gao, Zhitong and He, Xuming.*<br>
arXiv 2023. [[Paper](https://arxiv.org/abs/2306.11560)]                   
## Optimizing
## Data optimization techniques
## Data resampling 
116. **Repair: Removing representation bias by dataset resampling.**<br>
*Li, Yi and Vasconcelos, Nuno.*<br>
CVPR 2020. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_REPAIR_Removing_Representation_Bias_by_Dataset_Resampling_CVPR_2019_paper.pdf)]
117. **The class imbalance problem.**<br>
*Megahed, Fadel M and Chen, Ying-Ju and Megahed, Aly and Ong, Yuya and Altman, Naomi and Krzywinski, Martin.*<br>
Nature Methods 2021. [[Paper](https://www.nature.com/articles/s41592-021-01302-4)]
118. **Reslt: Residual learning for long-tailed recognition.**<br>
*Cui, Jiequan and Liu, Shu and Tian, Zhuotao and Zhong, Zhisheng and Jia, Jiaya.*<br>
TPAMI 2022. [[Paper](https://ieeexplore.ieee.org/abstract/document/9774921)]
119. **Batchbald: Efficient and diverse batch acquisition for deep bayesian active learning.**<br>
*Kirsch, Andreas and Van Amersfoort, Joost and Gal, Yarin.*<br>
NeurIPS 2020. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/95323660ed2124450caaac2c46b5ed90-Paper.pdf)]
120. **Online batch selection for faster training of neural networks.**<br>
*Loshchilov, Ilya and Hutter, Frank.*<br>
ICLR workshop track 2016. [[Paper](https://openreview.net/forum?id=r8lrkABJ7H8wknpYt5KB)]
121. **Learning from imbalanced data.**<br>
*He, Haibo and Garcia, Edwardo A.*<br>
TKDE 2009. [[Paper](https://ieeexplore.ieee.org/abstract/document/5128907)]
122. **Improving predictive inference under covariate shift by weighting the log-likelihood function.**<br>
*Shimodaira, Hidetoshi.*<br>
Journal of statistical planning and inference 2000. [[Paper](https://www.sciencedirect.com/science/article/pii/S0378375800001154?casa_token=rvwJ8e4TPt0AAAAA:TJUlHDHpcCd0-9xSlzt13K4hnlQQcF6Ed4e9JXzCBgAzQY8PPKah46j3f3QDZSedHP16vTFMVbw)]
123. **What is the effect of importance weighting in deep learning?.**<br>
*Byrd, Jonathon and Lipton, Zachary.*<br>
ICML 2019. [[Paper](https://proceedings.mlr.press/v97/byrd19a/byrd19a.pdf)]    
124. **Black-box importance sampling.**<br>
*Liu, Qiang and Lee, Jason.*<br>
AISTATS 2017. [[Paper](https://proceedings.mlr.press/v54/liu17b/liu17b.pdf)]  
125. **Not all samples are created equal: Deep learning with importance sampling.**<br>
*Katharopoulos, Angelos and Fleuret, Fran{\c{c}}ois.*<br>
ICML 2018. [[Paper](https://proceedings.mlr.press/v80/katharopoulos18a/katharopoulos18a.pdf)]  
126. **Gradient harmonized single-stage detector.**<br>
*Li, Buyu and Liu, Yu and Wang, Xiaogang.*<br>
AAAI 2019. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/4877)]
127. **Training deep models faster with robust, approximate importance sampling.**<br>
*Johnson, Tyler B and Guestrin, Carlos.*<br>
NeurIPS 2018. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2018/file/967990de5b3eac7b87d49a13c6834978-Paper.pdf)]  
128. **Accelerating deep learning by focusing on the biggest losers.**<br>
*Jiang, Angela H and Wong, Daniel L-K and Zhou, Giulio and Andersen, David G and Dean, Jeffrey and Ganger, Gregory R and Joshi, Gauri and Kaminksy, Michael and Kozuch, Michael and Lipton, Zachary C and Pillai, Padmanabhan.*<br>
arXiv 2020. [[Paper](https://arxiv.org/abs/1910.00762)]  
129. **Towards Understanding Deep Learning from Noisy Labels with Small-Loss Criterion.**<br>
*Gui, Xian-Jin and Wang, Wei and Tian, Zhang-Hao.*<br>
IJCAI 2021. [[Paper](https://www.ijcai.org/proceedings/2021/0340.pdf)]  
130. **Adaptiveface: Adaptive margin and sampling for face recognition.**<br>
*Liu, Hao and Zhu, Xiangyu and Lei, Zhen and Li, Stan Z.*<br>
CVPR 2019. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_AdaptiveFace_Adaptive_Margin_and_Sampling_for_Face_Recognition_CVPR_2019_paper.pdf)]                      
131. **Understanding the role of importance weighting for deep learning.**<br>
*Xu, Da and Ye, Yuting and Ruan, Chuanwei.*<br>
arXiv 2021. [[Paper](https://arxiv.org/abs/2103.15209)]  
132. **How to measure uncertainty in uncertainty sampling for active learning.**<br>
*Nguyen, Vu-Linh and Shaker, Mohammad Hossein and H{\"u}llermeier, Eyke.*<br>
Machine Learning 2022. [[Paper](https://link.springer.com/article/10.1007/s10994-021-06003-9)]  
133. **A survey on uncertainty estimation in deep learning classification systems from a Bayesian perspective.**<br>
*Mena, Jos{\'e} and Pujol, Oriol and Vitria, Jordi.*<br>
ACM Computing Surveys 2021. [[Paper](https://diposit.ub.edu/dspace/bitstream/2445/183476/1/714838.pdf)]  
134. **Uncertainty aware sampling framework of weak-label learning for histology image classification.**<br>
*Aljuhani, Asmaa and Casukhela, Ishya and Chan, Jany and Liebner, David and Machiraju, Raghu.*<br>
MICCAI 2022. [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-16434-7_36)]
135. **Optimal subsampling with influence functions.**<br>
*Ting, Daniel and Brochu, Eric.*<br>
NeurIPS 2018. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2018/file/57c0531e13f40b91b3b0f1a30b529a1d-Paper.pdf)]
136. **Background data resampling for outlier-aware classification.**<br>
*Li, Yi and Vasconcelos, Nuno.*<br>
CVPR 2020. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Background_Data_Resampling_for_Outlier-Aware_Classification_CVPR_2020_paper.pdf)]  
137. **Sentence-level resampling for named entity recognition.**<br>
*Wang, Xiaochen and Wang, Yue.*<br>
NAACL 2022. [[Paper](https://aclanthology.org/2022.naacl-main.156/)]  
138. **Undersampling near decision boundary for imbalance problems.**<br>
*Zhang, Jianjun and Wang, Ting and Ng, Wing WY and Zhang, Shuai and Nugent, Chris D.*<br>
ICMLC 2019. [[Paper](https://www.researchgate.net/profile/Jianjun-Zhang-4/publication/338453773_Undersampling_Near_Decision_Boundary_for_Imbalance_Problems/links/5e6982f992851c20f321f55b/Undersampling-Near-Decision-Boundary-for-Imbalance-Problems.pdf)]  
139. **Autosampling: Search for effective data sampling schedules.**<br>
*Sun, Ming and Dou, Haoxuan and Li, Baopu and Yan, Junjie and Ouyang, Wanli and Cui, Lei.*<br>
ICML 2021. [[Paper](https://proceedings.mlr.press/v139/sun21a/sun21a.pdf)]  
## Data augmentation
140. **Understanding data augmentation in neural machine translation: Two perspectives towards generalization.**<br>
*Li, Guanlin and Liu, Lemao and Huang, Guoping and Zhu, Conghui and Zhao, Tiejun.*<br>
EMNLP-IJCNLP 2019. [[Paper](https://aclanthology.org/D19-1570/)]
141. **Maximum-entropy adversarial data augmentation for improved generalization and robustness.**<br>
*Zhao, Long and Liu, Ting and Peng, Xi and Metaxas, Dimitris.*<br>
NeurIPS 2020. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/a5bfc9e07964f8dddeb95fc584cd965d-Paper.pdf)]
142. **Data augmentation can improve robustness.**<br>
*Rebuffi, Sylvestre-Alvise and Gowal, Sven and Calian, Dan Andrei and Stimberg, Florian and Wiles, Olivia and Mann, Timothy A.*<br>
NeurIPS 2021. [[Paper](https://proceedings.neurips.cc/paper/2021/file/fb4c48608ce8825b558ccf07169a3421-Paper.pdf)]
143. **Data augmentation alone can improve adversarial training.**<br>
*Li, Lin and Spratling, Michael W.*<br>
ICLR 2023. [[Paper](https://openreview.net/forum?id=y4uc4NtTWaq)]
144. **A survey on data augmentation for text classification.**<br>
*Bayer, Markus and Kaufhold, Marc-Andr{\'e} and Reuter, Christian.*<br>
ACM Computing Surveys 2022. [[Paper](https://dl.acm.org/doi/abs/10.1145/3544558)]
145. **A survey on image data augmentation for deep learning.**<br>
*Shorten, Connor and Khoshgoftaar, Taghi M.*<br>
Journal of big data 2019. [[Paper](https://journalofbigdata.springeropen.com/counter/pdf/10.1186/s40537-019-0197-0.pdf)]
146. **Data augmentation for deep graph learning: A survey.**<br>
*Ding, Kaize and Xu, Zhe and Tong, Hanghang and Liu, Huan.*<br>
ACM SIGKDD Explorations Newsletter 2022. [[Paper](https://dl.acm.org/doi/abs/10.1145/3575637.3575646)]
147. **Time Series Data Augmentation for Deep Learning: A Survey.**<br>
*Wen, Qingsong and Sun, Liang and Yang, Fan and Song, Xiaomin and Gao, Jingkun and Wang, Xue and Xu, Huan.*<br>
IJCAI 2021. [[Paper](https://www.ijcai.org/proceedings/2021/0631.pdf)]
### Augmentation target
148. **A survey on image data augmentation for deep learning.**<br>
*Shorten, Connor and Khoshgoftaar, Taghi M.*<br>
Journal of big data 2019. [[Paper](https://journalofbigdata.springeropen.com/counter/pdf/10.1186/s40537-019-0197-0.pdf)]
149. **Data augmentation approaches in natural language processing: A survey.**<br>
*Li, Bohan and Hou, Yutai and Che, Wanxiang.*<br>
Ai Open 2022. [[Paper](https://www.sciencedirect.com/science/article/pii/S2666651022000080)]
150. **Dataset augmentation in feature space.**<br>
*DeVries, Terrance and Taylor, Graham W.*<br>
ICLR Workshop track 2017. [[Paper](https://openreview.net/pdf?id=HJ9rLLcxg)]
151. **A simple feature augmentation for domain generalization.**<br>
*Li, Pan and Li, Da and Li, Wei and Gong, Shaogang and Fu, Yanwei and Hospedales, Timothy M.*<br>
ICCV 2021. [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Li_A_Simple_Feature_Augmentation_for_Domain_Generalization_ICCV_2021_paper.pdf)]
152. **Augmentation invariant and instance spreading feature for softmax embedding.**<br>
*Ye, Mang and Shen, Jianbing and Zhang, Xu and Yuen, Pong C and Chang, Shih-Fu.*<br>
TPAMI 2020. [[Paper](https://ieeexplore.ieee.org/abstract/document/9154587)]
153. **Feature space augmentation for long-tailed data.**<br>
*Chu, Peng and Bian, Xiao and Liu, Shaopeng and Ling, Haibin.*<br>
ECCV 2020. [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-58526-6_41)]
154. **Towards Deep Learning Models Resistant to Adversarial Attacks.**<br>
*Madry, Aleksander and Makelov, Aleksandar and Schmidt, Ludwig and Tsipras, Dimitris and Vladu, Adrian.*<br>
ICLR 2018. [[Paper](https://openreview.net/forum?id=rJzIBfZAb)]
155. **Recent Advances in Adversarial Training for Adversarial Robustness.**<br>
*Bai, Tao and Luo, Jinqi and Zhao, Jun and Wen, Bihan and Wang, Qian.*<br>
IJCAI 2021. [[Paper](https://www.ijcai.org/proceedings/2021/0591.pdf)]
156. **Graddiv: Adversarial robustness of randomized neural networks via gradient diversity regularization.**<br>
*Lee, Sungyoon and Kim, Hoki and Lee, Jaewook.*<br>
TPAMI 2022. [[Paper](https://ieeexplore.ieee.org/abstract/document/9761760)]
157. **Self-supervised label augmentation via input transformations.**<br>
*Lee, Hankook and Hwang, Sung Ju and Shin, Jinwoo.*<br>
ICML 2020. [[Paper](https://proceedings.mlr.press/v119/lee20c/lee20c.pdf)]
158. **Transductive label augmentation for improved deep network learning.**<br>
*Elezi, Ismail and Torcinovich, Alessandro and Vascon, Sebastiano and Pelillo, Marcello.*<br>
ICPR 2018. [[Paper](https://ieeexplore.ieee.org/abstract/document/8545524/)]
159. **Adversarial and isotropic gradient augmentation for image retrieval with text feedback.**<br>
*Huang, Fuxiang and Zhang, Lei and Zhou, Yuhang and Gao, Xinbo.*<br>
IEEE Transactions on Multimedia 2022. [[Paper](https://ieeexplore.ieee.org/abstract/document/9953564)]
### Augmentation strategy
160. **SMOTE: synthetic minority over-sampling technique}.**<br>
*Chawla, Nitesh V and Bowyer, Kevin W and Hall, Lawrence O and Kegelmeyer, W Philip.*<br>
Journal of artificial intelligence research 2002. [[Paper](https://www.jair.org/index.php/jair/article/view/10302)]
161. **A cost-sensitive deep learning-based approach for network traffic classification.**<br>
*Telikani, Akbar and Gandomi, Amir H and Choo, Kim-Kwang Raymond and Shen, Jun.*<br>
IEEE Transactions on Network and Service Management 2021. [[Paper](https://opus.lib.uts.edu.au/bitstream/10453/151147/2/368a20cf-97f8-48ab-8094-46fd31da71a9.pdf)]
162. **DeepSMOTE: Fusing deep learning and SMOTE for imbalanced data.**<br>
*Dablain, Damien and Krawczyk, Bartosz and Chawla, Nitesh V.*<br>
TNNLS 2022. [[Paper](https://ieeexplore.ieee.org/abstract/document/9694621)]
163. **mixup: Beyond Empirical Risk Minimization.**<br>
*Zhang, Hongyi and Cisse, Moustapha and Dauphin, Yann N and Lopez-Paz, David.*<br>
ICLR 2018. [[Paper](https://openreview.net/forum?id=r1Ddp1-Rb&;noteId=r1Ddp1-Rb)]
164. **Manifold mixup: Better representations by interpolating hidden states.**<br>
*Verma, Vikas and Lamb, Alex and Beckham, Christopher and Najafi, Amir and Mitliagkas, Ioannis and Lopez-Paz, David and Bengio, Yoshua.*<br>
ICML 2019. [[Paper](https://proceedings.mlr.press/v97/verma19a/verma19a.pdf)]
165. **Generative adversarial nets.**<br>
*Goodfellow, Ian and Pouget-Abadie, Jean and Mirza, Mehdi and Xu, Bing and Warde-Farley, David and Ozair, Sherjil and Courville, Aaron and Bengio, Yoshua.*<br>
NeurIPS 2014. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf)]
166. **A review on generative adversarial networks: Algorithms, theory, and applications.**<br>
*Gui, Jie and Sun, Zhenan and Wen, Yonggang and Tao, Dacheng and Ye, Jieping.*<br>
TKDE 2021. [[Paper](https://ieeexplore.ieee.org/abstract/document/9625798)]
167. **BAGAN: Data Augmentation with Balancing GAN.**<br>
*Mariani, Giovanni and Scheidegger, Florian and Istrate, Roxana and Bekas, Costas and Malossi, Cristiano.*<br>
ICML 2018. [[Paper](https://research.ibm.com/publications/bagan-data-augmentation-with-balancing-gan)]
168. **Auggan: Cross domain adaptation with gan-based data augmentation.**<br>
*Huang, Sheng-Wei and Lin, Che-Tsung and Chen, Shu-Ping and Wu, Yen-Yi and Hsu, Po-Hao and Lai, Shang-Hong.*<br>
ECCV 2018. [[Paper](https://openaccess.thecvf.com/content_ECCV_2018/html/Sheng-Wei_Huang_AugGAN_Cross_Domain_ECCV_2018_paper.html)]
169. **TS-GAN: Time-series GAN for Sensor-based Health Data Augmentation.**<br>
*Yang, Zhenyu and Li, Yantao and Zhou, Gang.*<br>
ACM Transactions on Computing for Healthcare 2023. [[Paper](https://dl.acm.org/doi/abs/10.1145/3583593)]
170. **Diffusion models: A comprehensive survey of methods and applications.**<br>
*Yang, Ling and Zhang, Zhilong and Song, Yang and Hong, Shenda and Xu, Runsheng and Zhao, Yue and Zhang, Wentao and Cui, Bin and Yang, Ming-Hsuan.*<br>
ACM Computing Surveys 2022. [[Paper](https://dl.acm.org/doi/abs/10.1145/3626235)]
171. **Multimodal Data Augmentation for Image Captioning using Diffusion Models.**<br>
*Xiao, Changrong and Xu, Sean Xin and Zhang, Kunpeng.*<br>
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.01855)]
172. **Diversify your vision datasets with automatic diffusion-based augmentation.**<br>
*Dunlap, Lisa and Umino, Alyssa and Zhang, Han and Yang, Jiezhi and Gonzalez, Joseph E and Darrell, Trevor.*<br>
arXiv 2023. [[Paper](https://arxiv.org/abs/2305.16289)]
173. **An Extensive Exploration of Back-Translation in 60 Languages.**<br>
*McNamee, Paul and Duh, Kevin.*<br>
Findings of ACL 2023. [[Paper](https://aclanthology.org/2023.findings-acl.518/)]
174. **I2t2i: Learning text to image synthesis with textual data augmentation.**<br>
*Dong, Hao and Zhang, Jingqing and McIlwraith, Douglas and Guo, Yike.*<br>
ICIP 2017. [[Paper](https://ieeexplore.ieee.org/abstract/document/8296635)]
175. **Set-level Guidance Attack: Boosting Adversarial Transferability of Vision-Language Pre-training Models.**<br>
*Lu, Dong and Wang, Zhiqiang and Wang, Teng and Guan, Weili and Gao, Hongchang and Zheng, Feng.*<br>
ICCV 2023. [[Paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Lu_Set-level_Guidance_Attack_Boosting_Adversarial_Transferability_of_Vision-Language_Pre-training_Models_ICCV_2023_paper.pdf)]
176. **TTIDA: Controllable Generative Data Augmentation via Text-to-Text and Text-to-Image Models.**<br>
*Yin, Yuwei and Kaddour, Jean and Zhang, Xiang and Nie, Yixin and Liu, Zhenguang and Kong, Lingpeng and Liu, Qi.*<br>
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.08821)]
177. **Combining Adversaries with Anti-adversaries in Training.**<br>
*Zhou, Xiaoling and Yang, Nan and Wu, Ou.*<br>
AAAI 2023. [[Paper](https://arxiv.org/abs/2304.12550)]
178. **Improving generalization via uncertainty driven perturbations.**<br>
*Pagliardini, Matteo and Manunza, Gilberto and Jaggi, Martin and Jordan, Michael I and Chavdarova, Tatjana.*<br>
arXiv 2022. [[Paper](https://arxiv.org/abs/2202.05737)]
179. **Randaugment: Practical automated data augmentation with a reduced search space.**<br>
*Cubuk, Ekin D and Zoph, Barret and Shlens, Jonathon and Le, Quoc V.*<br>
CVPR workshops 2020. [[Paper](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w40/Cubuk_Randaugment_Practical_Automated_Data_Augmentation_With_a_Reduced_Search_Space_CVPRW_2020_paper.pdf)]
180. **Metamixup: Learning adaptive interpolation policy of mixup with metalearning.**<br>
*Mai, Zhijun and Hu, Guosheng and Chen, Dexiong and Shen, Fumin and Shen, Heng Tao.*<br>
TNNLS 2021. [[Paper](https://ieeexplore.ieee.org/abstract/document/9366422)]
181. **Automatic data augmentation via deep reinforcement learning for effective kidney tumor segmentation.**<br>
*Qin, Tiexin and Wang, Ziyuan and He, Kelei and Shi, Yinghuan and Gao, Yang and Shen, Dinggang.*<br>
ICASSP 2020. [[Paper](https://ieeexplore.ieee.org/abstract/document/9053403)]
182. **Augmentation strategies for learning with noisy labels.**<br>
*Nishi, Kento and Ding, Yi and Rich, Alex and Hollerer, Tobias.*<br>
CVPR 2021. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Nishi_Augmentation_Strategies_for_Learning_With_Noisy_Labels_CVPR_2021_paper.pdf)]
183. **Differentiable automatic data augmentation.**<br>
*Li, Yonggang and Hu, Guosheng and Wang, Yongtao and Hospedales, Timothy and Robertson, Neil M and Yang, Yongxin.*<br>
ECCV 2020. [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-58542-6_35)]
184. **Implicit semantic data augmentation for deep networks.**<br>
*Wang, Yulin and Pan, Xuran and Song, Shiji and Zhang, Hong and Huang, Gao and Wu, Cheng.*<br>
NeurIPS 2019. [[Paper](https://proceedings.neurips.cc/paper/2019/file/15f99f2165aa8c86c9dface16fefd281-Paper.pdf)]
185. **Imagine by reasoning: A reasoning-based implicit semantic data augmentation for long-tailed classification.**<br>
*Chen, Xiaohua and Zhou, Yucan and Wu, Dayan and Zhang, Wanqian and Zhou, Yu and Li, Bo and Wang, Weiping.*<br>
AAAI 2022. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/19912)]
186. **Implicit Counterfactual Data Augmentation for Deep Neural Networks.**<br>
*Zhou, Xiaoling and Wu, Ou.*<br>
arXiv 2023. [[Paper](https://arxiv.org/abs/2304.13431)]
187. **On feature normalization and data augmentation.**<br>
*Li, Boyi and Wu, Felix and Lim, Ser-Nam and Belongie, Serge and Weinberger, Kilian Q.*<br>
CVPR 2021. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_On_Feature_Normalization_and_Data_Augmentation_CVPR_2021_paper.pdf)]
188. **Implicit rugosity regularization via data augmentation.**<br>
*LeJeune, Daniel and Balestriero, Randall and Javadi, Hamid and Baraniuk, Richard G.*<br>
arXiv 2019. [[Paper](https://arxiv.org/abs/1905.11639)]
189. **Mixup as locally linear out-of-manifold regularization.**<br>
*Guo, Hongyu and Mao, Yongyi and Zhang, Richong.*<br>
AAAI 2019. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/4256)]
190. **Avoiding overfitting: A survey on regularization methods for convolutional neural networks.**<br>
*Santos, Claudio Filipi Gon{\c{c}}alves Dos and Papa, Jo{\~a}o Paulo.*<br>
ACM Computing Surveys 2022. [[Paper](https://dl.acm.org/doi/full/10.1145/3510413)]
191. **The good, the bad and the ugly sides of data augmentation: An implicit spectral regularization perspective.**<br>
*Lin, Chi-Heng and Kaushik, Chiraag and Dyer, Eva L and Muthukumar, Vidya.*<br>
arXiv 2022. [[Paper](https://arxiv.org/abs/2210.05021)]
192. **A group-theoretic framework for data augmentation.**<br>
*Chen, Shuxiao and Dobriban, Edgar and Lee, Jane H.*<br>
The Journal of Machine Learning Research 2020. [[Paper](https://dl.acm.org/doi/abs/10.5555/3455716.3455961)]
## Data perturbation
193. **Compensation learning.**<br>
*Yao, Rujing and Wu, Ou.*<br>
arXiv 2021. [[Paper](https://arxiv.org/abs/2107.11921)]
### Perturbation target
194. **Learn2perturb: an end-to-end feature perturbation learning to improve adversarial robustness.**<br>
*Jeddi, Ahmadreza and Shafiee, Mohammad Javad and Karg, Michelle and Scharfenberger, Christian and Wong, Alexander.*<br>
CVPR 2020. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jeddi_Learn2Perturb_An_End-to-End_Feature_Perturbation_Learning_to_Improve_Adversarial_Robustness_CVPR_2020_paper.pdf)]
195. **Encoding robustness to image style via adversarial feature perturbations.**<br>
*Shu, Manli and Wu, Zuxuan and Goldblum, Micah and Goldstein, Tom.*<br>
NeurIPS 2021. [[Paper](https://proceedings.neurips.cc/paper/2021/file/ec20019911a77ad39d023710be68aaa1-Paper.pdf)]
196. **Logit perturbation.**<br>
*Li, Mengyang and Su, Fengguang and Wu, Ou and Zhang, Ji.*<br>
AAAI 2022. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/20024)]
197. **Long-tail learning via logit adjustment.**<br>
*Menon, Aditya Krishna and Jayasumana, Sadeep and Rawat, Ankit Singh and Jain, Himanshu and Veit, Andreas and Kumar, Sanjiv.*<br>
ICLR 2021. [[Paper](https://openreview.net/pdf?id=37nvvqkCo5)]
198. **Learning imbalanced datasets with label-distribution-aware margin loss.**<br>
*Cao, Kaidi and Wei, Colin and Gaidon, Adrien and Arechiga, Nikos and Ma, Tengyu.*<br>
NeurIPS 2019. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/621461af90cadfdaf0e8d4cc25129f91-Paper.pdf)]
199. **Implicit semantic data augmentation for deep networks.**<br>
*Wang, Yulin and Pan, Xuran and Song, Shiji and Zhang, Hong and Huang, Gao and Wu, Cheng.*<br>
NeurIPS 2019. [[Paper](https://proceedings.neurips.cc/paper/2019/file/15f99f2165aa8c86c9dface16fefd281-Paper.pdf)]
200. **Class-Level Logit Perturbation.**<br>
*Li, Mengyang and Su, Fengguang and Wu, Ou and Zhang, Ji.*<br>
TNNLS 2023. [[Paper](https://ieeexplore.ieee.org/abstract/document/10130785/)]
201. **Rethinking the inception architecture for computer vision.**<br>
*Szegedy, Christian and Vanhoucke, Vincent and Ioffe, Sergey and Shlens, Jon and Wojna, Zbigniew.*<br>
CVPR 2016. [[Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)]
202. **Delving deep into label smoothing.**<br>
*Zhang, Chang-Bin and Jiang, Peng-Tao and Hou, Qibin and Wei, Yunchao and Han, Qi and Li, Zhen and Cheng, Ming-Ming.*<br>
TIP 2021. [[Paper](https://ieeexplore.ieee.org/abstract/document/9464693)]
203. **Adversarial robustness via label-smoothing.**<br>
*Goibert, Morgane and Dohmatob, Elvis.*<br>
arXiv 2019. [[Paper](https://arxiv.org/abs/1906.11567)]
204. **From label smoothing to label relaxation.**<br>
*Lienen, Julian and H{\"u}llermeier, Eyke.*<br>
AAAI 2021. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/17041)]
205. **Anticorrelated noise injection for improved generalization.**<br>
*Orvieto, Antonio and Kersting, Hans and Proske, Frank and Bach, Francis and Lucchi, Aurelien.*<br>
ICML 2022. [[Paper](https://proceedings.mlr.press/v162/orvieto22a/orvieto22a.pdf)]
206. **Adversarial weight perturbation helps robust generalization.**<br>
*Wu, Dongxian and Xia, Shu-Tao and Wang, Yisen.*<br>
NeurIPS 2020. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/1ef91c212e30e14bf125e9374262401f-Paper.pdf)]
207. **Reinforcement learning with perturbed rewards.**<br>
*Wang, Jingkang and Liu, Yang and Li, Bo.*<br>
AAAI 2020. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/6086)]
### Perturbation direction
208. **Implicit semantic data augmentation for deep networks.**<br>
*Wang, Yulin and Pan, Xuran and Song, Shiji and Zhang, Hong and Huang, Gao and Wu, Cheng.*<br>
NeurIPS 2019. [[Paper](https://proceedings.neurips.cc/paper/2019/file/15f99f2165aa8c86c9dface16fefd281-Paper.pdf)]
209. **Combining Adversaries with Anti-adversaries in Training.**<br>
*Zhou, Xiaoling and Yang, Nan and Wu, Ou.*<br>
AAAI 2023. [[Paper](https://arxiv.org/abs/2304.12550)]
210. **Training deep neural networks on noisy labels with bootstrapping.**<br>
*Reed, Scott and Lee, Honglak and Anguelov, Dragomir and Szegedy, Christian and Erhan, Dumitru and Rabinovich, Andrew.*<br>
ICLR workshop 2015. [[Paper](https://imgtec.eetrend.com/sites/imgtec.eetrend.com/files/201709/blog/10381-29627-deep.pdf)]
211. **Logit perturbation.**<br>
*Li, Mengyang and Su, Fengguang and Wu, Ou and Zhang, Ji.*<br>
AAAI 2022. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/20024)]
### Perturbation granularity
212. **Universal adversarial training with class-wise perturbations.**<br>
*Benz, Philipp and Zhang, Chaoning and Karjauv, Adil and Kweon, In So.*<br>
ICME 2021. [[Paper](https://ieeexplore.ieee.org/abstract/document/9428419/)]
213. **Balancing Logit Variation for Long-tailed Semantic Segmentation.**<br>
*Wang, Yuchao and Fei, Jingjing and Wang, Haochen and Li, Wei and Bao, Tianpeng and Wu, Liwei and Zhao, Rui and Shen, Yujun.*<br>
CVPR 2023. [[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Balancing_Logit_Variation_for_Long-Tailed_Semantic_Segmentation_CVPR_2023_paper.pdf)]
214. **Universal adversarial training.**<br>
*Shafahi, Ali and Najibi, Mahyar and Xu, Zheng and Dickerson, John and Davis, Larry S and Goldstein, Tom.*<br>
AAAI 2020. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/6017)]
215. **Universal adversarial perturbations.**<br>
*Moosavi-Dezfooli, Seyed-Mohsen and Fawzi, Alhussein and Fawzi, Omar and Frossard, Pascal.*<br>
CVPR 2017. [[Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Moosavi-Dezfooli_Universal_Adversarial_Perturbations_CVPR_2017_paper.pdf)]
216. **Distribution-balanced loss for multi-label classification in long-tailed datasets.**<br>
*Wu, Tong and Huang, Qingqiu and Liu, Ziwei and Wang, Yu and Lin, Dahua.*<br>
ECCV 2020. [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-58548-8_10)]
### Assignment manner
217. **Transferable adversarial perturbations.**<br>
*Zhou, Wen and Hou, Xin and Chen, Yongjun and Tang, Mengyun and Huang, Xiangqi and Gan, Xiang and Yang, Yong.*<br>
ECCV 2018. [[Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Bruce_Hou_Transferable_Adversarial_Perturbations_ECCV_2018_paper.pdf)]
218. **Sparse adversarial perturbations for videos.**<br>
*Wei, Xingxing and Zhu, Jun and Yuan, Sha and Su, Hang.*<br>
AAAI 2019. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/4927)]
219. **Investigating annotation noise for named entity recognition.**<br>
*Zhu, Yu and Ye, Yingchun and Li, Mengyang and Zhang, Ji and Wu, Ou.*<br>
Neural Computing and Applications 2023. [[Paper](https://link.springer.com/article/10.1007/s00521-022-07733-0)]
220. **A simple framework for contrastive learning of visual representations.**<br>
*Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey.*<br>
ICML 2020. [[Paper](https://proceedings.mlr.press/v119/chen20j.html)]
221. **A self-supervised approach for adversarial robustness.**<br>
*Naseer, Muzammal and Khan, Salman and Hayat, Munawar and Khan, Fahad Shahbaz and Porikli, Fatih.*<br>
CVPR 2020. [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Naseer_A_Self-supervised_Approach_for_Adversarial_Robustness_CVPR_2020_paper.pdf)]
222. **GANSER: A self-supervised data augmentation framework for EEG-based emotion recognition.**<br>
*Zhang, Zhi and Zhong, Sheng-hua and Liu, Yan.*<br>
IEEE Transactions on Affective Computing 2022. [[Paper](https://ieeexplore.ieee.org/abstract/document/9763358/)]
223. **Metasaug: Meta semantic augmentation for long-tailed visual recognition.**<br>
*Li, Shuang and Gong, Kaixiong and Liu, Chi Harold and Wang, Yulin and Qiao, Feng and Cheng, Xinjing.*<br>
CVPR 2021. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_MetaSAug_Meta_Semantic_Augmentation_for_Long-Tailed_Visual_Recognition_CVPR_2021_paper.pdf)]
224. **Uncertainty-guided model generalization to unseen domains.**<br>
*Qiao, Fengchun and Peng, Xi.*<br>
CVPR 2021. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Qiao_Uncertainty-Guided_Model_Generalization_to_Unseen_Domains_CVPR_2021_paper.pdf)]
225. **Autoaugment: Learning augmentation policies from data.**<br>
*Cubuk, Ekin D and Zoph, Barret and Mane, Dandelion and Vasudevan, Vijay and Le, Quoc V.*<br>
CVPR 2019. [[Paper](https://research.google/pubs/pub47890/)]
226. **Automatically Learning Data Augmentation Policies for Dialogue Tasks.**<br>
*Niu, Tong and Bansal, Mohit.*<br>
EMNLP 2019. [[Paper](https://aclanthology.org/D19-1132/)]
227. **Deep reinforcement adversarial learning against botnet evasion attacks.**<br>
*Apruzzese, Giovanni and Andreolini, Mauro and Marchetti, Mirco and Venturi, Andrea and Colajanni, Michele.*<br>
IEEE Transactions on Network and Service Management 2020. [[Paper](https://ieeexplore.ieee.org/abstract/document/9226405)]
228. **Adversarial reinforced instruction attacker for robust vision-language navigation.**<br>
*Lin, Bingqian and Zhu, Yi and Long, Yanxin and Liang, Xiaodan and Ye, Qixiang and Lin, Liang.*<br>
TPAMI 2021. [[Paper](https://ieeexplore.ieee.org/abstract/document/9488322)]
## Data weighting
### Weighting granularity
229. **Denoising implicit feedback for recommendation.**<br>
*Wang, Wenjie and Feng, Fuli and He, Xiangnan and Nie, Liqiang and Chua, Tat-Seng.*<br>
WSDM 2021. [[Paper](https://dl.acm.org/doi/abs/10.1145/3437963.3441800)]
230. **Superloss: A generic loss for robust curriculum learning.**<br>
*Castells, Thibault and Weinzaepfel, Philippe and Revaud, Jerome.*<br>
NeurIPS 2020. [[Paper](https://proceedings.neurips.cc/paper/2020/file/2cfa8f9e50e0f510ede9d12338a5f564-Paper.pdf)]
231. **Class-balanced loss based on effective number of samples.**<br>
*Cui, Yin and Jia, Menglin and Lin, Tsung-Yi and Song, Yang and Belongie, Serge.*<br>
CVPR 2019. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf)]
232. **Distribution alignment: A unified framework for long-tail visual recognition.**<br>
*Zhang, Songyang and Li, Zeming and Yan, Shipeng and He, Xuming and Sun, Jian.*<br>
CVPR 2021. [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Distribution_Alignment_A_Unified_Framework_for_Long-Tail_Visual_Recognition_CVPR_2021_paper.pdf)]
233. **Dynamically weighted balanced loss: class imbalanced learning and confidence calibration of deep neural networks.**<br>
*Fernando, K Ruwani M and Tsokos, Chris P.*<br>
TNNLS 2021. [[Paper](https://ieeexplore.ieee.org/abstract/document/9324926)]
234. **Just train twice: Improving group robustness without training group information.**<br>
*Liu, Evan Z and Haghgoo, Behzad and Chen, Annie S and Raghunathan, Aditi and Koh, Pang Wei and Sagawa, Shiori and Liang, Percy and Finn, Chelsea.*<br>
ICML 2021. [[Paper](https://proceedings.mlr.press/v139/liu21f/liu21f.pdf)]
235. **Geometry-aware Instance-reweighted Adversarial Training.**<br>
*Zhang, Jingfeng and Zhu, Jianing and Niu, Gang and Han, Bo and Sugiyama, Masashi and Kankanhalli, Mohan.*<br>
ICLR 2021. [[Paper](https://openreview.net/forum?id=iAX0l6Cz8ub)]
236. **Credit card fraud detection: a realistic modeling and a novel learning strategy.**<br>
*Dal Pozzolo, Andrea and Boracchi, Giacomo and Caelen, Olivier and Alippi, Cesare and Bontempi, Gianluca.*<br>
TNNLS 2017. [[Paper](https://ieeexplore.ieee.org/abstract/document/8038008)]
237. **Cost-sensitive portfolio selection via deep reinforcement learning.**<br>
*Zhang, Yifan and Zhao, Peilin and Wu, Qingyao and Li, Bin and Huang, Junzhou and Tan, Mingkui.*<br>
TKDE 2020. [[Paper](https://ieeexplore.ieee.org/abstract/document/9031418)]
238. **Integrating TANBN with cost sensitive classification algorithm for imbalanced data in medical diagnosis.**<br>
*Gan, Dan and Shen, Jiang and An, Bang and Xu, Man and Liu, Na.*<br>
Computers & Industrial Engineering 2020. [[Paper](https://www.sciencedirect.com/science/article/pii/S0360835219307351?casa_token=97voI68djkMAAAAA:cRy98l9KsYxqelE8TlpklR7e7RcZD2dz9VvkF0Eg6FvwXAwvrCjJKfTbyzREOuY-TtDae5Hroiw)]
239. **FORML: Learning to Reweight Data for Fairness.**<br>
*Yan, Bobby and Seto, Skyler and Apostoloff, Nicholas.*<br>
arXiv 2022. [[Paper](https://arxiv.org/abs/2202.01719)]
240. **Fairness in graph mining: A survey.**<br>
*Dong, Yushun and Ma, Jing and Wang, Song and Chen, Chen and Li, Jundong.*<br>
TKDE 2023. [[Paper](https://ieeexplore.ieee.org/abstract/document/10097603)]
### Weighting factors
241. **Curriculum learning.**<br>
*Bengio, Yoshua and Louradour, J{\'e}r{\^o}me and Collobert, Ronan and Weston, Jason.*<br>
ICML 2009. [[Paper](https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/15972/Bengio%2C%202009%20Curriculum%20Learning.pdf?sequence=1&isAllowed=y)]
242. **Self-paced learning for latent variable models.**<br>
*Kumar, M and Packer, Benjamin and Koller, Daphne.*<br>
NeurIPS 2010. [[Paper](https://proceedings.neurips.cc/paper/2010/file/e57c6b956a6521b28495f2886ca0977a-Paper.pdf)]
243. **Easy samples first: Self-paced reranking for zero-example multimedia search.**<br>
*Jiang, Lu and Meng, Deyu and Mitamura, Teruko and Hauptmann, Alexander G.*<br>
ACM MM 2014. [[Paper](https://dl.acm.org/doi/abs/10.1145/2647868.2654918)]
244. **Self-paced learning with diversity.**<br>
*Jiang, Lu and Meng, Deyu and Yu, Shoou-I and Lan, Zhenzhong and Shan, Shiguang and Hauptmann, Alexander.*<br>
NeurIPS 2014. [[Paper](https://proceedings.neurips.cc/paper/2014/file/c60d060b946d6dd6145dcbad5c4ccf6f-Paper.pdf)]
245. **A self-paced multiple-instance learning framework for co-saliency detection.**<br>
*Zhang, Dingwen and Meng, Deyu and Li, Chao and Jiang, Lu and Zhao, Qian and Han, Junwei.*<br>
ICCV 2015. [[Paper](https://openaccess.thecvf.com/content_iccv_2015/papers/Zhang_A_Self-Paced_Multiple-Instance_ICCV_2015_paper.pdf)]
246. **Curriculum learning: A survey.**<br>
*Soviany, Petru and Ionescu, Radu Tudor and Rota, Paolo and Sebe, Nicu.*<br>
IJCV 2022. [[Paper](https://link.springer.com/article/10.1007/s11263-022-01611-x)]
247. **Focal loss for dense object detection.**<br>
*Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr.*<br>
ICCV 2017. [[Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)]
248. **Geometry-aware Instance-reweighted Adversarial Training.**<br>
*Zhang, Jingfeng and Zhu, Jianing and Niu, Gang and Han, Bo and Sugiyama, Masashi and Kankanhalli, Mohan.*<br>
ICLR 2021. [[Paper](https://openreview.net/forum?id=iAX0l6Cz8ub)]
249. **LOW: Training deep neural networks by learning optimal sample weights.**<br>
*Santiago, Carlos and Barata, Catarina and Sasdelli, Michele and Carneiro, Gustavo and Nascimento, Jacinto C.*<br>
Pattern Recognition 2021. [[Paper](https://www.sciencedirect.com/science/article/pii/S0031320320303885?casa_token=4OHT8wlvtroAAAAA:PRX8tFrNiPLvbPzQ7Fgsu9k-gUmqdNCePi0JdJJQzHzQarTjTeeo3MsnAsrc4lDwRSlqdKDuZLQ)]
250. **Curriculum Learning with Diversity for Supervised Computer Vision Tasks.**<br>
*Soviany, Petru.*<br>
ICML Workshop 2020. [[Paper](https://openreview.net/forum?id=WH27bUkkzj)]
251. **Which Samples Should Be Learned First: Easy or Hard?**<br>
*Which Samples Should Be Learned First: Easy or Hard?.*<br>
TNNLS 2023. [[Paper](https://ieeexplore.ieee.org/abstract/document/10155763)]
252. **Metacleaner: Learning to hallucinate clean representations for noisy-labeled visual recognition.**<br>
*Zhang, Weihe and Wang, Yali and Qiao, Yu.*<br>
CVPR 2019. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_MetaCleaner_Learning_to_Hallucinate_Clean_Representations_for_Noisy-Labeled_Visual_Recognition_CVPR_2019_paper.pdf)]
253. **Confident learning: Estimating uncertainty in dataset labels.**<br>
*Northcutt, Curtis and Jiang, Lu and Chuang, Isaac.*<br>
Journal of Artificial Intelligence Research 2021. [[Paper](https://www.jair.org/index.php/jair/article/view/12125)]
### Assignment manners
254. **Class-balanced loss based on effective number of samples.**<br>
*Cui, Yin and Jia, Menglin and Lin, Tsung-Yi and Song, Yang and Belongie, Serge.*<br>
CVPR 2019. [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf)]
255. **Focal loss for dense object detection.**<br>
*Lin, Tsung-Yi and Goyal, Priya and Girshick, Ross and He, Kaiming and Doll{\'a}r, Piotr.*<br>
ICCV 2017. [[Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)]
256. **Umix: Improving importance weighting for subpopulation shift via uncertainty-aware mixup.**<br>
*Han, Zongbo and Liang, Zhipeng and Yang, Fan and Liu, Liu and Li, Lanqing and Bian, Yatao and Zhao, Peilin and Wu, Bingzhe and Zhang, Changqing and Yao, Jianhua.*<br>
NeurIPS 2022. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/f593c9c251d4d7cf14d4ab9861dfb7eb-Paper-Conference.pdf)]
257. **Classification with noisy labels by importance reweighting.**<br>
*Liu, Tongliang and Tao, Dacheng.*<br>
TPAMI 2015. [[Paper](https://ieeexplore.ieee.org/abstract/document/7159100)]
258. **Self-paced learning for latent variable models.**<br>
*Kumar, M and Packer, Benjamin and Koller, Daphne.*<br>
NeurIPS 2010. [[Paper](https://proceedings.neurips.cc/paper/2010/file/e57c6b956a6521b28495f2886ca0977a-Paper.pdf)]  
259. **Self-paced learning: An implicit regularization perspective.**<br>
*Fan, Yanbo and He, Ran and Liang, Jian and Hu, Baogang.*<br>
AAAI 2017. [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/10809)]
260. **Adversarial reweighting for partial domain adaptation.**<br>
*Gu, Xiang and Yu, Xi and Sun, Jian and Xu, Zongben.*<br>
NeurIPS 2021. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2021/file/7ce3284b743aefde80ffd9aec500e085-Paper.pdf)]
261. **Reweighting Augmented Samples by Minimizing the Maximal Expected Loss.**<br>
*Yi, Mingyang and Hou, Lu and Shang, Lifeng and Jiang, Xin and Liu, Qun and Ma, Zhi-Ming.*<br>
ICLR 2021. [[Paper](https://openreview.net/forum?id=9G5MIc-goqB)]
262. **Learning to reweight examples for robust deep learning.**<br>
*Ren, Mengye and Zeng, Wenyuan and Yang, Bin and Urtasun, Raquel.*<br>
ICML 2018. [[Paper](https://proceedings.mlr.press/v80/ren18a/ren18a.pdf)]
263. **Meta-weight-net: Learning an explicit mapping for sample weighting.**<br>
*Shu, Jun and Xie, Qi and Yi, Lixuan and Zhao, Qian and Zhou, Sanping and Xu, Zongben and Meng, Deyu.*<br>
NeurIPS 2019. [[Paper](https://proceedings.neurips.cc/paper_files/paper/2019/file/e58cc5ca94270acaceed13bc82dfedf7-Paper.pdf)]
