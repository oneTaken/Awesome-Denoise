长期招聘AI画质算法实习与社招。
如果你对以下一些方向感兴趣，或者有过相关经历，可以欢迎联系pxp.cv@qq.com
+ 图像超分辨率重建、降噪、HDR重建等
+ 基于Nerf等场景重建算法
+ AI 算法量化与onnx部署

---
# Awesome-Denoise 

There are three main factors to divide these papers into different catrgories to have a better idea.  
Sometimes raw domain denoising papers would use some ISP to convert to sRGB domain, So use Both to cover this situation.  
Sometimes video denoising papers degrade to burst denoising, even single image denoising, always use Video tag to cover this situation.  

* Color Space
  * RGB
  * Raw
  * Both

* Image Kind
  * Single
  * Burst
  * Video

* Noise Model  
  * AWGN(Additive White Gaussian Noise model)  
  * PG(Posion Gaussian noise model)  
  * GAN(Gan based noise model)  
  * Real(camera or dlsr devices real noise model)  
  * Prior
    * Low Rank
    * Sparsity
    * self similarity

## benchmark dataset  

* SIDD, CVPR 2018, citation 256
  * [A High-Quality Denoising Dataset for Smartphone Cameras](https://openaccess.thecvf.com/content_cvpr_2018/papers/Abdelhamed_A_High-Quality_Denoising_CVPR_2018_paper.pdf)
  * [Matlab](https://github.com/AbdoKamel/sidd-ground-truth-image-estimation)
* RENOIR, JVCIR 2018, citation 106
  * [RENOIR–A dataset for real low-light image noise reduction](https://arxiv.org/pdf/1409.8230.pdf)
  * [broken dataset link](http://adrianbarburesearch.blogspot.com/p/renoir-dataset.html)
* PolyU, arxiv 2018, citation 108
  * [Real-world Noisy Image Denoising: A New Benchmark](https://arxiv.org/pdf/1804.02603.pdf)
  * [Matlab](https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset)
* SID, CVPR 2018, citation 595
  * [Learning to see in the dark](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Learning_to_See_CVPR_2018_paper.pdf)
  * [Tensorflow](https://github.com/cchen156/Learning-to-See-in-the-Dark)
* DND, CVPR 2017, citation 296
  * [Benchmarking Denoising Algorithms with Real Photographs](https://openaccess.thecvf.com/content_cvpr_2017/papers/Plotz_Benchmarking_Denoising_Algorithms_CVPR_2017_paper.pdf)
  * [homepage](https://noise.visinf.tu-darmstadt.de/)
* NaM, CVPR 2016, citation 148
  * [A Holistic Approach to Cross-Channel Image Noise Modeling and its Application to Image Denoising](https://openaccess.thecvf.com/content_cvpr_2016/papers/Nam_A_Holistic_Approach_CVPR_2016_paper.pdf)|


# self-supervised denoising
video denoising
+ [Unsupervised deep video denoising](http://openaccess.thecvf.com/content/ICCV2021/html/Sheth_Unsupervised_Deep_Video_Denoising_ICCV_2021_paper.html)
  + ICCV 2021, UDVD
+ [Recurrent Self-Supervised Video Denoising with Denser
Receptive Field](https://arxiv.org/pdf/2308.03608.pdf)
  + ACM MM 2023, [code](https://github.com/Wang-XIaoDingdd/RDRF)

image denoising

|Index|Year|Pub|Title|cite|
|:---:|:---:|:---:|:---:|:---:|
|1|2018|ICML|[Noise2Noise: Learning image restoration without clean data](https://arxiv.org/pdf/1803.04189.pdf)|1236|
|2|2019|CVPR|[Noise2void-learning denoising from single noisy images](http://openaccess.thecvf.com/content_CVPR_2019/html/Krull_Noise2Void_-_Learning_Denoising_From_Single_Noisy_Images_CVPR_2019_paper.html)|748|
|3|2019|ICML|[Noise2self: Blind denoising by self-supervision](http://proceedings.mlr.press/v97/batson19a.html)|441|
|4|2019|NeurIPS|[High-quality self-supervised deep image denoising](https://proceedings.neurips.cc/paper/8920-high-quality-self-supervised-deep-image-denoising)|247|
|5|2019|arxiv|[Unsupervised image noise modeling with self-consistent GAN](https://arxiv.org/pdf/1906.05762.pdf)|13|
|6|2020|Frontiers in Computer Science|[Probabilistic noise2void: Unsupervised content-aware denoising](https://www.frontiersin.org/articles/10.3389/fcomp.2020.00005/full)|119|
|7|2020|TIP|[Noisy-as-clean: Learning self-supervised denoising from corrupted image](https://arxiv.org/pdf/1906.06878.pdf)|112|
|8|2020|CVPR|[Self2self with dropout: Learning self-supervised denoising from single image](http://openaccess.thecvf.com/content_CVPR_2020/html/Quan_Self2Self_With_Dropout_Learning_Self-Supervised_Denoising_From_Single_Image_CVPR_2020_paper.html)|201|
|9|2020|CVPR|[Noisier2noise: Learning to denoise from unpaired noisy data](http://openaccess.thecvf.com/content_CVPR_2020/html/Moran_Noisier2Noise_Learning_to_Denoise_From_Unpaired_Noisy_Data_CVPR_2020_paper.html)|125|
|10|2020|NeurIPS|[Noise2Same: Optimizing a self-supervised bound for image denoising](https://proceedings.neurips.cc/paper/2020/hash/ea6b2efbdd4255a9f1b3bbc6399b58f4-Abstract.html)|57|
|11|2021|NeurIPS|[Noise2score: tweedie's approach to self-supervised image denoising without clean images](https://proceedings.neurips.cc/paper/2021/hash/077b83af57538aa183971a2fe0971ec1-Abstract.html)|32|
|12|2021|CVPR|[Neighbor2neighbor: Self-supervised denoising from single noisy images](http://openaccess.thecvf.com/content/CVPR2021/html/Huang_Neighbor2Neighbor_Self-Supervised_Denoising_From_Single_Noisy_Images_CVPR_2021_paper.html)|135|
|13|2021|CVPR|[Recorrupted-to-recorrupted: unsupervised deep learning for image denoising](http://openaccess.thecvf.com/content/CVPR2021/html/Pang_Recorrupted-to-Recorrupted_Unsupervised_Deep_Learning_for_Image_Denoising_CVPR_2021_paper.html)|85|
|14|2022|TIP|Neighbor2Neighbor: A Self-Supervised Framework for Deep Image Denoising|7|
|15|2022|CVPR|[Ap-bsn: Self-supervised denoising for real-world images via asymmetric pd and blind-spot network](http://openaccess.thecvf.com/content/CVPR2022/html/Lee_AP-BSN_Self-Supervised_Denoising_for_Real-World_Images_via_Asymmetric_PD_and_CVPR_2022_paper.html)|27|
|16|2022|CVPR|[CVF-SID: Cyclic multi-variate function for self-supervised image denoising by disentangling noise from image](http://openaccess.thecvf.com/content/CVPR2022/html/Neshatavar_CVF-SID_Cyclic_Multi-Variate_Function_for_Self-Supervised_Image_Denoising_by_Disentangling_CVPR_2022_paper.html)|20|
|17|2022|CVPR|[Self-supervised deep image restoration via adaptive stochastic gradient langevin dynamics](http://openaccess.thecvf.com/content/CVPR2022/html/Wang_Self-Supervised_Deep_Image_Restoration_via_Adaptive_Stochastic_Gradient_Langevin_Dynamics_CVPR_2022_paper.html)|7|
|18|2022|CVPR|[Noise distribution adaptive self-supervised image denoising using tweedie distribution and score matching](http://openaccess.thecvf.com/content/CVPR2022/html/Kim_Noise_Distribution_Adaptive_Self-Supervised_Image_Denoising_Using_Tweedie_Distribution_and_CVPR_2022_paper.html)|5|
|19|2022|CVPR|[Blind2unblind: Self-supervised image denoising with visible blind spots](http://openaccess.thecvf.com/content/CVPR2022/html/Wang_Blind2Unblind_Self-Supervised_Image_Denoising_With_Visible_Blind_Spots_CVPR_2022_paper.html)|29|
|20|2022|CVPR|[Idr: Self-supervised image denoising via iterative data refinement](http://openaccess.thecvf.com/content/CVPR2022/html/Zhang_IDR_Self-Supervised_Image_Denoising_via_Iterative_Data_Refinement_CVPR_2022_paper.html)|22|
|21|2023|CVPR|[Spatially Adaptive Self-Supervised Learning for Real-World Image Denoising](http://openaccess.thecvf.com/content/CVPR2023/html/Li_Spatially_Adaptive_Self-Supervised_Learning_for_Real-World_Image_Denoising_CVPR_2023_paper.html)|1|
|22|2023|CVPR|[LG-BPN: Local and Global Blind-Patch Network for Self-Supervised Real-World Denoising](http://openaccess.thecvf.com/content/CVPR2023/html/Wang_LG-BPN_Local_and_Global_Blind-Patch_Network_for_Self-Supervised_Real-World_Denoising_CVPR_2023_paper.html)|0|
|23|2023|CVPR|[Zero-Shot Noise2Noise: Efficient Image Denoising Without Any Data](https://openaccess.thecvf.com/content/CVPR2023/html/Mansour_Zero-Shot_Noise2Noise_Efficient_Image_Denoising_Without_Any_Data_CVPR_2023_paper.html)|1|
|24|2023|CVPR|[Patch-Craft Self-Supervised Training for Correlated Image Denoising](https://openaccess.thecvf.com/content/CVPR2023/html/Vaksman_Patch-Craft_Self-Supervised_Training_for_Correlated_Image_Denoising_CVPR_2023_paper.html)|
|25|2023|arxiv|[Unleashing the Power of Self-Supervised Image Denoising: A Comprehensive Review](https://arxiv.org/pdf/2308.00247.pdf)|
|26|2023|ICCV|[Random Sub-Samples Generation for Self-Supervised Real Image Denoising](https://arxiv.org/pdf/2307.16825.pdf)|
|27|2023|ICCV|[Score Priors Guided Deep Variational Inference for Unsupervised Real-World Single Image Denoising](https://arxiv.org/pdf/2308.04682.pdf)|
|28|2023|ICCV|[Unsupervised Image Denoising in Real-World Scenarios via Self-Collaboration Parallel Generative Adversarial Branches](https://arxiv.org/pdf/2308.06776.pdf)|

# by year
## 2020

|Pub|Title|Code|Cite|
|:---:|:---:|:---:|:---:|
|TIP|[Noisy-As-Clean: Learning Self-supervised Denoising from Corrupted Image](http://mftp.mmcheng.net/Papers/20TIP_NAC.pdf)|[Pytorch](https://github.com/csjunxu/Noisy-As-Clean-TIP2020)|47|
|TIP|[Blind universal Bayesian image denoising with Gaussian noise level learning](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9024220)|-|43|
|TIP|[Learning Deformable Kernels for Image and Video Denoising](https://arxiv.org/pdf/1904.06903.pdf)|-|24|
|TIP|Learning Spatial and Spatio-Temporal Pixel Aggregations for Image and Video Denoising|-|10|
|TIP|[Deep Graph-Convolutional Image Denoising](https://arxiv.org/pdf/1907.08448.pdf)|-|64|
|TIP|[NLH : A Blind Pixel-level Non-local Method for Real-world Image Denoising](https://arxiv.org/pdf/1906.06834.pdf)|-|34|
|TIP|[Image Denoising via Sequential Ensemble Learning](https://cpb-us-w2.wpmucdn.com/blog.nus.edu.sg/dist/8/10877/files/2020/03/TIP2020_ensemble.pdf)|-|13|
|TIP|[Connecting Image Denoising and High-Level Vision Tasks via Deep Learning](https://arxiv.org/pdf/1809.01826.pdf)|-|70|
|CVPR|[Memory-Efficient Hierarchical Neural Architecture Search for Image Denoising](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Memory-Efficient_Hierarchical_Neural_Architecture_Search_for_Image_Denoising_CVPR_2020_paper.pdf)|-|33|
|CVPR|[A Physics-based Noise Formation Model for Extreme Low-light Raw Denoising](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_A_Physics-Based_Noise_Formation_Model_for_Extreme_Low-Light_Raw_Denoising_CVPR_2020_paper.pdf)|[Pytorch](https://github.com/Vandermode/ELD)|50|
|CVPR|[Supervised Raw Video Denoising With a Benchmark Dataset on Dynamic Scenes](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yue_Supervised_Raw_Video_Denoising_With_a_Benchmark_Dataset_on_Dynamic_CVPR_2020_paper.pdf)|[Pytorch](https://github.com/cao-cong/RViDeNet)|26|Both|Video|Real|
|CVPR|[Transfer Learning From Synthetic to Real-Noise Denoising With Adaptive Instance Normalization](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kim_Transfer_Learning_From_Synthetic_to_Real-Noise_Denoising_With_Adaptive_Instance_CVPR_2020_paper.pdf)|-|60|
|CVPR|[Self2Self With Dropout: Learning Self-Supervised Denoising From Single Image](https://openaccess.thecvf.com/content_CVPR_2020/papers/Quan_Self2Self_With_Dropout_Learning_Self-Supervised_Denoising_From_Single_Image_CVPR_2020_paper.pdf)|-|73|
|CVPR|[Noisier2Noise: Learning to Denoise From Unpaired Noisy Data](https://openaccess.thecvf.com/content_CVPR_2020/papers/Moran_Noisier2Noise_Learning_to_Denoise_From_Unpaired_Noisy_Data_CVPR_2020_paper.pdf)|-|40|
|CVPR|[Joint Demosaicing and Denoising With Self Guidance](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Joint_Demosaicing_and_Denoising_With_Self_Guidance_CVPR_2020_paper.pdf)|-|26|
|CVPR|[FastDVDnet: Towards Real-Time Deep Video Denoising Without Flow Estimation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Tassano_FastDVDnet_Towards_Real-Time_Deep_Video_Denoising_Without_Flow_Estimation_CVPR_2020_paper.pdf)|-|72|RGB|Video|AWGN|
|CVPR|[CycleISP: Real Image Restoration via Improved Data Synthesis](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zamir_CycleISP_Real_Image_Restoration_via_Improved_Data_Synthesis_CVPR_2020_paper.pdf)|[Pytorch](https://github.com/swz30/CycleISP)|93|
|CVPR|[Basis Prediction Networks for Effective Burst Denoising With Large Kernels](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xia_Basis_Prediction_Networks_for_Effective_Burst_Denoising_With_Large_Kernels_CVPR_2020_paper.pdf)|-|18|
|CVPR|[Superkernel Neural Architecture Search for Image Denoising](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Mozejko_Superkernel_Neural_Architecture_Search_for_Image_Denoising_CVPRW_2020_paper.pdf)|-|5|
|ECCV|[Spatial-Adaptive Network for Single Image Denoising](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123750171.pdf)|-|34|
|ECCV|[A Decoupled Learning Scheme for Real-world Burst Denoising from Raw Images](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700154.pdf)|-|3|
|ECCV|[Burst Denoising via Temporally Shifted Wavelet Transforms](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580239.pdf)|-|0|
|ECCV|[Unpaired Learning of Deep Image Denoising](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490341.pdf)|[Pytorch](https://github.com/XHWXD/DBSN)|24|
|ECCV|[Dual Adversarial Network: Toward Real-world Noise Removal and Noise Generation](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550035.pdf)|[Pytorch](https://github.com/zsyOAOA/DANet)|39|
|ECCV|[Learning Camera-Aware Noise Models](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690341.pdf)|[Pytorch](https://github.com/arcchang1236/CA-NoiseGAN)|9|
|ECCV|[Practical Deep Raw Image Denoising on Mobile Devices](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510001.pdf)|[MegEngine](https://github.com/megvii-research/PMRID)|15|Raw|Single|PG|
|ECCV|[Reconstructing the Noise Manifold for Image Denoising](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540596.pdf)|-|2|
|NN|[Deep Learning on Image Denoising : An Overview](https://arxiv.org/pdf/1912.13171.pdf)|-|247|
|WACV|[Identifying recurring patterns with deep neural networks for natural image denoising](http://openaccess.thecvf.com/content_WACV_2020/papers/Xia_Identifying_Recurring_Patterns_with_Deep_Neural_Networks_for_Natural_Image_WACV_2020_paper.pdf)|-|11|
|ICASSP|[Attention Mechanism Enhanced Kernel Prediction Networks for Denoising of Burst Images](https://arxiv.org/pdf/1910.08313.pdf)|[Pytorch](https://github.com/z-bingo/Attention-Mechanism-Enhanced-KPN)|4|
|Arxiv|[Low-light Image Restoration with Short- and Long-exposure Raw Pairs](https://arxiv.org/pdf/2007.00199.pdf)|-|6|

## 2019  

|Pub|Title|Code|Cite|
|:---:|:---:|:---:|:---:|
|TIP|[Optimal combination of image denoisers](https://arxiv.org/pdf/1711.06712.pdf)|-|13|
|TIP|[High ISO JPEG Image Denoising by Deep Fusion of Collaborative and Convolutional Filtering](https://sci-hub.se/https://ieeexplore.ieee.org/abstract/document/8684332/)|-|6|
|TIP|[Texture variation adaptive image denoising with nonlocal PCA](https://arxiv.org/pdf/1810.11282.pdf)|-|11|
|TIP|[Color Image and Multispectral Image Denoising Using Block Diagonal Representation](https://arxiv.org/pdf/1902.03954.pdf)|-|7|
|TIP|Tchebichef and Adaptive Steerable-Based Total Variation Model for Image Denoising|-|23|
|TIP|[Iterative Joint Image Demosaicking and Denoising Using a Residual Denoising Network](https://arxiv.org/pdf/1807.06403.pdf)|-|55|
|TIP|Content-Adaptive Noise Estimation for Color Images with Cross-Channel Noise Modeling|-|4|
|TPAMI|[Real-world Image Denoising with Deep Boosting](https://sci-hub.se/https://ieeexplore.ieee.org/abstract/document/8733117/)|[Tensorflow](https://github.com/ngchc/deepBoosting)|29|
|JVCIR|Vst-net: Variance-stabilizing transformation inspired network for poisson denoising|[Matlab](https://github.com/yqx7150/VST-Net)|14|
|NIPS|[Variational Denoising Network: Toward Blind Noise Modeling and Removal](https://papers.nips.cc/paper/8446-variational-denoising-network-toward-blind-noise-modeling-and-removal.pdf)|-|110|
|NIPS|[High-Quality Self-Supervised Deep Image Denoising](http://papers.nips.cc/paper/8920-high-quality-self-supervised-deep-image-denoising.pdf)|-|138|
|ICML|[Noise2Self: Blind Denoising by Self-Supervision](https://arxiv.org/pdf/1901.11365.pdf)|[Pytorch](https://github.com/czbiohub/noise2self)|244|
|ICML|[Plug-and-play methods provably converge with properly trained denoisers](https://arxiv.org/pdf/1905.05406.pdf)|-|125|
|CVPR|[Unsupervised Domain Adaptation for ToF Data Denoising with Adversarial Learning](https://openaccess.thecvf.com/content_CVPR_2019/papers/Agresti_Unsupervised_Domain_Adaptation_for_ToF_Data_Denoising_With_Adversarial_Learning_CVPR_2019_paper.pdf)|-|26|
|CVPR|[Robust Subspace Clustering with Independent and Piecewise Identically Distributed Noise Modeling](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Robust_Subspace_Clustering_With_Independent_and_Piecewise_Identically_Distributed_Noise_CVPR_2019_paper.pdf)|-|15|
|CVPR|[Toward convolutional blind denoising of real photographs](http://openaccess.thecvf.com/content_CVPR_2019/papers/Guo_Toward_Convolutional_Blind_Denoising_of_Real_Photographs_CVPR_2019_paper.pdf)|[Matlab](https://github.com/GuoShi28/CBDNet)|458|
|CVPR|[FOCNet: A Fractional Optimal Control Network for Image Denoising](http://openaccess.thecvf.com/content_CVPR_2019/papers/Jia_FOCNet_A_Fractional_Optimal_Control_Network_for_Image_Denoising_CVPR_2019_paper.pdf)|-|62|
|CVPR|[Noise2void-learning denoising from single noisy images](http://openaccess.thecvf.com/content_CVPR_2019/papers/Krull_Noise2Void_-_Learning_Denoising_From_Single_Noisy_Images_CVPR_2019_paper.pdf)|-|406|
|CVPR|[Unprocessing images for learned raw denoising](http://openaccess.thecvf.com/content_CVPR_2019/papers/Brooks_Unprocessing_Images_for_Learned_Raw_Denoising_CVPR_2019_paper.pdf)|-|186|
|CVPR|[Training deep learning based image denoisers from undersampled measurements without ground truth and without image prior](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhussip_Training_Deep_Learning_Based_Image_Denoisers_From_Undersampled_Measurements_Without_CVPR_2019_paper.pdf)|-|28|
|CVPR|[Model-blind video denoising via frame-to-frame training](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ehret_Model-Blind_Video_Denoising_via_Frame-To-Frame_Training_CVPR_2019_paper.pdf)|[other](https://github.com/tehret/blind-denoising)|44|
|ICCV|[Self-Guided Network for Fast Image Denoising](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gu_Self-Guided_Network_for_Fast_Image_Denoising_ICCV_2019_paper.pdf)|-|78|
|ICCV|[Noise flow: Noise modeling with conditional normalizing flows](https://openaccess.thecvf.com/content_ICCV_2019/papers/Abdelhamed_Noise_Flow_Noise_Modeling_With_Conditional_Normalizing_Flows_ICCV_2019_paper.pdf)|-|74|
|ICCV|[Joint Demosaicking and Denoising by Fine-Tuning of Bursts of Raw Images](https://openaccess.thecvf.com/content_ICCV_2019/papers/Ehret_Joint_Demosaicking_and_Denoising_by_Fine-Tuning_of_Bursts_of_Raw_ICCV_2019_paper.pdf)|-|34|
|ICCV|[Fully Convolutional Pixel Adaptive Image Denoiser](https://openaccess.thecvf.com/content_ICCV_2019/papers/Cha_Fully_Convolutional_Pixel_Adaptive_Image_Denoiser_ICCV_2019_paper.pdf)|[Keras](https://github.com/csm9493/FC-AIDE-Keras)|27|
|ICCV|[Enhancing Low Light Videos by Exploring High Sensitivity Camera Noise](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Enhancing_Low_Light_Videos_by_Exploring_High_Sensitivity_Camera_Noise_ICCV_2019_paper.pdf)|-|14|
|ICCV|[CIIDefence: Defeating Adversarial Attacks by Fusing Class-Specific Image Inpainting and Image Denoising](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gupta_CIIDefence_Defeating_Adversarial_Attacks_by_Fusing_Class-Specific_Image_Inpainting_and_ICCV_2019_paper.pdf)|-|21|
|ICCV|[Real Image Denoising with Feature Attention](https://arxiv.org/pdf/1904.07396.pdf)|-|192|
|CVPRW|[GRDN:Grouped Residual Dense Network for Real Image Denoising and GAN-based Real-world Noise Modeling](http://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Kim_GRDNGrouped_Residual_Dense_Network_for_Real_Image_Denoising_and_GAN-Based_CVPRW_2019_paper.pdf)|-|65|
|CVPRW|[Learning raw image denoising with bayer pattern unification and bayer preserving augmentation](https://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Liu_Learning_Raw_Image_Denoising_With_Bayer_Pattern_Unification_and_Bayer_CVPRW_2019_paper.pdf)|-|29|
|CVPRW|[Deep iterative down-up CNN for image denoising](http://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Yu_Deep_Iterative_Down-Up_CNN_for_Image_Denoising_CVPRW_2019_paper.pdf)|-|69|
|CVPRW|[Densely Connected Hierarchical Network for Image Denoising](https://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Park_Densely_Connected_Hierarchical_Network_for_Image_Denoising_CVPRW_2019_paper.pdf)|-|55|
|CVPRW|[ViDeNN: Deep Blind Video Denoising](http://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Claus_ViDeNN_Deep_Blind_Video_Denoising_CVPRW_2019_paper.pdf)|-|42|
|CVPRW|[Real Photographs Denoising With Noise Domain Adaptation and Attentive Generative Adversarial Network](http://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Lin_Real_Photographs_Denoising_With_Noise_Domain_Adaptation_and_Attentive_Generative_CVPRW_2019_paper.pdf)|-|15|
|CVPRW|[Learning Deep Image Priors for Blind Image Denoising](http://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Hou_Learning_Deep_Image_Priors_for_Blind_Image_Denoising_CVPRW_2019_paper.pdf)|-|4|
|ICIP|[DVDnet: A fast network for deep video denoising](https://arxiv.org/pdf/1906.11890.pdf)|[Pytorch](https://github.com/m-tassano/dvdnet)|45|RGB|Video|AWGN|
|ICIP|[Multi-kernel prediction networks for denoising of burst images](https://arxiv.org/pdf/1902.05392.pdf)|-|17|
|ICIP|A non-local cnn for video denoising|-|31|
|AAAI|Adaptation Strategies for Applying AWGN-based Denoiser to Realistic Noise|-|4|
|arxiv|[When AWGN-based Denoiser Meets Real Noises](https://arxiv.org/pdf/1904.03485.pdf)|[Pytorch](https://github.com/yzhouas/PD-Denoising-pytorch)|29|
|arxiv|[Generating training data for denoising real rgb images via camera pipeline simulation](https://arxiv.org/pdf/1904.08825.pdf)|-|19|
|arxiv|[Learning Deformable Kernels for Image and Video Denoising](https://arxiv.org/pdf/1904.06903.pdf)|-|24|
|arxiv|[Gan2gan: Generative noise learning for blind image denoising with single noisy images](https://arxiv.org/pdf/1905.10488.pdf)|-|12|

## 2018  

|Pub|Title|Code|Cite|
|:---:|:---:|:---:|:---:|
|TIP|Weighted Tensor Rank-1 Decomposition for Nonlocal Image Denoising|-|19|
|TIP|Towards Optimal Denoising of Image Contrast|-|8|
|TIP|[Time-of-Flight Range Measurement in Low- sensing Environment : Noise Analysis and Complex-domain Non-local Denoising](https://www.researchgate.net/profile/Mihail_Georgiev4/publication/323233188_Time-of-Flight_Range_Measurement_in_Low-Sensing_Environment_Noise_Analysis_and_Complex-Domain_Non-Local_Denoising/links/5b2373750f7e9b0e374893a7/Time-of-Flight-Range-Measurement-in-Low-Sensing-Environment-Noise-Analysis-and-Complex-Domain-Non-Local-Denoising.pdf)|-|10|
|TIP|[Statistical Nearest Neighbors for Image Denoising](https://research.nvidia.com/sites/default/files/pubs/2018-09_Statistical-Nearest-Neighbors/Statistical%20Nearest%20Neighbors%20for%20Image%20Denoising.pdf)|-|29|
|TIP|[Joint Denoising / Compression of Image Contours via Shape Prior and Context Tree](https://arxiv.org/pdf/1705.00268.pdf)|-|5|
|TIP|[Image Restoration by Iterative Denoising and Backward Projections](https://arxiv.org/pdf/1710.06647.pdf)|-|110|
|TIP|Corrupted reference image quality assessment of denoised images|-|11|
|TIP|[FFDNet: Toward a fast and flexible solution for CNN-based image denoising](https://arxiv.org/pdf/1710.04026.pdf)|[Matlab](https://github.com/cszn/FFDNet)|1103|
|TIP|[External prior guided internal prior learning for real-world noisy image denoising](https://arxiv.org/pdf/1705.04505.pdf)|-|92|
|TIP|[Class-aware fully convolutional Gaussian and Poisson denoising](https://arxiv.org/pdf/1808.06562.pdf)|[Tensorflow](https://github.com/TalRemez/deep_class_aware_denoising)|54|
|TIP|[VIDOSAT: High-dimensional sparsifying transform learning for online video denoising](https://arxiv.org/pdf/1710.00947.pdf)|-|23|
|TIP|[Effective and fast estimation for image sensor noise via constrained weighted least squares](https://www.researchgate.net/profile/Jiantao_Zhou/publication/323563338_Effective_and_Fast_Estimation_for_Image_Sensor_Noise_Via_Constrained_Weighted_Least_Squares/links/5acdcaa6a6fdcc87840afac1/Effective-and-Fast-Estimation-for-Image-Sensor-Noise-Via-Constrained-Weighted-Least-Squares.pdf)|-|20|
|ToG|Denoising with kernel prediction and asymmetric loss functions|-|106|
|TMM|Gradient prior-aided cnn denoiser with separable convolution-based optimization of feature dimension|-|22|
|NIPS|[Training deep learning based denoisers without ground truth data](https://papers.nips.cc/paper/7587-training-deep-learning-based-denoisers-without-ground-truth-data.pdf)|-|75|
|ICML|[Noise2Noise: Learning Image Restoration without Clean Data](https://arxiv.org/pdf/1803.04189.pdf)|-|758|
|CVPR|[Burst denoising with kernel prediction networks](http://openaccess.thecvf.com/content_cvpr_2018/papers/Mildenhall_Burst_Denoising_With_CVPR_2018_paper.pdf)|-|224|
|CVPR|[Image Blind Denoising With Generative Adversarial Network Based Noise Modeling](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Image_Blind_Denoising_CVPR_2018_paper.pdf)|-|352|
|CVPR|[Universal Denoising Networks : A Novel CNN Architecture for Image Denoising](http://openaccess.thecvf.com/content_cvpr_2018/papers/Lefkimmiatis_Universal_Denoising_Networks_CVPR_2018_paper.pdf)|[Matlab](https://github.com/cig-skoltech/UDNet)|209|
|ECCV|[Deep burst denoising](http://openaccess.thecvf.com/content_ECCV_2018/papers/Clement_Godard_Deep_Burst_Denoising_ECCV_2018_paper.pdf)|-|74|
|ECCV|[Deep boosting for image denoising](http://openaccess.thecvf.com/content_ECCV_2018/papers/Chang_Chen_Deep_Boosting_for_ECCV_2018_paper.pdf)|-|50|
|ECCV|[A trilateral weighted sparse coding scheme for real-world image denoising](http://openaccess.thecvf.com/content_ECCV_2018/papers/XU_JUN_A_Trilateral_Weighted_ECCV_2018_paper.pdf)|-|180|
|ECCV|[Deep image demosaicking using a cascade of convolutional residual denoising networks](https://openaccess.thecvf.com/content_ECCV_2018/papers/Filippos_Kokkinos_Deep_Image_Demosaicking_ECCV_2018_paper.pdf)|-|68|
|IJCAI|[Connecting image denoising and high-level vision tasks via deep learning](https://arxiv.org/pdf/1809.01826.pdf)|-|70|
|IJCAI|[When image denoising meets high-level vision tasks: A deep learning approach](https://arxiv.org/pdf/1706.04284.pdf)|-|160|
|JVCIR|[RENOIR–A dataset for real low-light image noise reduction](https://arxiv.org/pdf/1409.8230.pdf)|-|106|
|TCI|[Convolutional neural networks for noniterative reconstruction of compressively sensed images](https://arxiv.org/pdf/1708.04669.pdf)|-|83|
|ACCV|[Dn-resnet: Efficient deep residual network for image denoising](https://arxiv.org/pdf/1810.06766.pdf)|-|22|
|ICIP|[Image Denoising for Image Retrieval by Cascading a Deep Quality Assessment Network](http://www.ee.iisc.ac.in/new/people/faculty/soma.biswas/Papers/biju_icip2018.pdf)|-|9|
|arxiv|[Correction by projection: Denoising images with generative adversarial networks](https://arxiv.org/pdf/1803.04477.pdf)|-|47|
|arxiv|[Non-local video denoising by CNN](https://arxiv.org/pdf/1811.12758.pdf)|[Pytorch](https://github.com/axeldavy/vnlnet)|31|
|arxiv|[Iterative residual network for deep joint image demosaicking and denoising](https://arxiv.org/pdf/1807.06403.pdf)|-|9|
|arxiv|[Fully convolutional pixel adaptive image denoiser](https://arxiv.org/pdf/1807.07569.pdf)|-|27|
|arxiv|[Fast, trainable, multiscale denoising](https://arxiv.org/pdf/1802.06130.pdf)|-|6|
|arxiv|[Deep learning for image denoising: a survey](https://arxiv.org/pdf/1810.05052.pdf)|-|90|

## 2017  

|Publ|Title|Code|Cite|
|:---:|:---:|:---:|:---:|
|TIP|[Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising](https://arxiv.org/pdf/1608.03981.pdf)|-|4387|
|TIP|Improved Denoising via Poisson Mixture Modeling of Image Sensor Noise|-|29|
|TIP|Reweighted Low-Rank Matrix Analysis with Structural Smoothness for Image Denoising|-|40|
|TIP|Category-specific object image denoising|-|31|
|TIP|[Affine Non-Local Means Image Denoising](https://repositori.upf.edu/bitstream/handle/10230/37095/ballester_trans26_affi.pdf?sequence=1&isAllowed=y)|-|39|
|CVPR|[Image Denoising via CNNs: An Adversarial Approach](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Divakar_Image_Denoising_via_CVPR_2017_paper.pdf)|-|71|
|CVPR|[Non-local color image denoising with convolutional neural networks](http://openaccess.thecvf.com/content_cvpr_2017/papers/Lefkimmiatis_Non-Local_Color_Image_CVPR_2017_paper.pdf)|-|274|
|CVPR|[Learning Deep CNN Denoiser Prior for Image Restoration](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhang_Learning_Deep_CNN_CVPR_2017_paper.pdf)|-|1277|
|ICCV|[Learning Proximal Operators : Using Denoising Networks for Regularizing Inverse Imaging Problems](https://openaccess.thecvf.com/content_ICCV_2017/papers/Meinhardt_Learning_Proximal_Operators_ICCV_2017_paper.pdf)|-|246|
|ICCV|[Multi-channel Weighted Nuclear Norm Minimization for Real Color Image Denoising](https://openaccess.thecvf.com/content_ICCV_2017/papers/Xu_Multi-Channel_Weighted_Nuclear_ICCV_2017_paper.pdf)|-|230|
|ICCV|[Joint Adaptive Sparsity and Low-Rankness on the Fly: An Online Tensor Reconstruction Scheme for Video Denoising](https://openaccess.thecvf.com/content_ICCV_2017/papers/Wen_Joint_Adaptive_Sparsity_ICCV_2017_paper.pdf)|-|40|
|ICCV|[Blob Reconstruction Using Unilateral Second Order Gaussian Kernels with Application to High-ISO Long-Exposure Image Denoising](https://openaccess.thecvf.com/content_ICCV_2017/papers/Wang_Blob_Reconstruction_Using_ICCV_2017_paper.pdf)|-|10|
|ICIP|[Image denoising using group sparsity residual and external nonlocal self-similarity prior](https://arxiv.org/pdf/1701.00723.pdf)|-|7|
|arxiv|[Block-matching convolutional neural network for image denoising](https://arxiv.org/pdf/1704.00524.pdf)|-|50|
|arxiv|[Learning pixel-distribution prior with wider convolution for image denoising](https://arxiv.org/pdf/1707.09135.pdf)|[Matlab](https://github.com/cswin/WIN)|19|
|arxiv|[Chaining identity mapping modules for image denoising](https://arxiv.org/pdf/1712.02933.pdf)|-|12|
|ICTAI|[Dilated deep residual network for image denoising](https://arxiv.org/pdf/1708.05473.pdf)|-|73|

## before 2017  

|Year|Publication|Title|Code|Citation|
|:---:|:---:|:---:|:---:|:---:|
|2016|CVPR|[Deep Gaussian conditional random field network: A model-based deep network for discriminative denoising](https://openaccess.thecvf.com/content_cvpr_2016/papers/Vemulapalli_Deep_Gaussian_Conditional_CVPR_2016_paper.pdf)|-|68|
|2016|CVPR|[From Noise Modeling to Blind Image Denoising](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhu_From_Noise_Modeling_CVPR_2016_paper.pdf)|-|67|
|2016|TIP|Patch-based video denoising with optical flow estimation|-|99|
|2016|ToG|Deep joint demosaicking and denoising|-|336|
|2016|ICASSP|Fast depth image denoising and enhancement using a deep convolutional network|-|62|
|2015|ICCV|[An efficient statistical method for image noise level estimation](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Chen_An_Efficient_Statistical_ICCV_2015_paper.pdf)|-|184|
|2015|TIP|Image-specific prior adaptation for denoising|-|19|
|2015|IPOL|[The noise clinic: a blind image denoising algorithm](http://www.ipol.im/pub/art/2015/125/article_lr.pdf)|-|112|
|2014|TIP|Practical signal-dependent noise parameter estimation from a single noisy image|-|86|
|2014|-|[Photon, Poisson Noise](http://people.csail.mit.edu/hasinoff/pubs/hasinoff-photon-2011-preprint.pdf)|-|107|
|2012|CVPR|[Image denoising: Can plain neural networks compete with BM3D?](https://hcburger.com/files/neuraldenoising.pdf)|-|1246|
|2012|ICIP|The dominance of Poisson noise in color digital cameras|-|29|
|2009|SP|[Clipped noisy images: Heteroskedastic modeling and practical denoising](https://www.researchgate.net/profile/Alessandro_Foi/publication/220227880_Clipped_noisy_images_Heteroskedastic_modeling_and_practical_denoising/links/5b7d594c299bf1d5a71c4b11/Clipped-noisy-images-Heteroskedastic-modeling-and-practical-denoising.pdf)|-|129|
|2008|TIP|[Practical Poissonian-Gaussian noise modeling and fitting for single-image raw-data](https://core.ac.uk/download/pdf/194121585.pdf)|Matlab|723|
|2007|TIP|[Image denoising by sparse 3-D transform-domain collaborative filtering](http://web.eecs.utk.edu/~hqi/ece692/references/noise-BM3D-tip07.pdf)|-|7357|
|2007|TPAMI|[Automatic estimation and removal of noise from a single image](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.228.3525&rep=rep1&type=pdf)|-|599|
|2005|CVPR|[A non-local algorithm for image denoising](http://audio.rightmark.org/lukin/msu/NonLocal.pdf)|-|7477|
|2019|Books|CMOS: Circuit Design, Layout, and Simulation: Forth Edition|-|5390|
|2018|Books|Denoising of photographic images and video: fundamentals, open challenges and new trends|-|14|
