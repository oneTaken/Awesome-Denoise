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

* SIDD, CVPR 2018, citation 101
  * [A High-Quality Denoising Dataset for Smartphone Cameras](https://openaccess.thecvf.com/content_cvpr_2018/papers/Abdelhamed_A_High-Quality_Denoising_CVPR_2018_paper.pdf)
  * [Matlab](https://github.com/AbdoKamel/sidd-ground-truth-image-estimation)
* RENOIR, JVCIR 2018, citation 57
  * [RENOIR–A dataset for real low-light image noise reduction](https://arxiv.org/pdf/1409.8230.pdf)
  * [broken dataset link](http://adrianbarburesearch.blogspot.com/p/renoir-dataset.html)
* PolyU, arxiv 2018, citation 55
  * [Real-world Noisy Image Denoising: A New Benchmark](https://arxiv.org/pdf/1804.02603.pdf)
  * [Matlab](https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset)
* SID, CVPR 2018, citation 253
  * [Learning to see in the dark](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Learning_to_See_CVPR_2018_paper.pdf)
  * [Tensorflow](https://github.com/cchen156/Learning-to-See-in-the-Dark)
* DND, CVPR 2017, citation 164
  * [Benchmarking Denoising Algorithms with Real Photographs](https://openaccess.thecvf.com/content_cvpr_2017/papers/Plotz_Benchmarking_Denoising_Algorithms_CVPR_2017_paper.pdf)
  * [homepage](https://noise.visinf.tu-darmstadt.de/)
* NaM, CVPR 2016, citation 84
  * [A Holistic Approach to Cross-Channel Image Noise Modeling and its Application to Image Denoising](https://openaccess.thecvf.com/content_cvpr_2016/papers/Nam_A_Holistic_Approach_CVPR_2016_paper.pdf)|

## 2020

|Pub|Title|Color|Image|Noise|Code|Cite|
|:---:|:---:|:---:|:---:|:----:|:----:|:-----:|
|TIP|[Noisy-As-Clean: Learning Self-supervised Denoising from Corrupted Image](http://mftp.mmcheng.net/Papers/20TIP_NAC.pdf)|[Pytorch](https://github.com/csjunxu/Noisy-As-Clean-TIP2020)|0|
|TIP|Blind universal Bayesian image denoising with Gaussian noise level learning|-|8|
|TIP|[Learning Deformable Kernels for Image and Video Denoising](https://arxiv.org/pdf/1904.06903.pdf)|-|6|
|TIP|Learning Spatial and Spatio-Temporal Pixel Aggregations for Image and Video Denoising|-|0|
|TIP|[Deep Graph-Convolutional Image Denoising](https://arxiv.org/pdf/1907.08448.pdf)|-|10|
|TIP|[NLH : A Blind Pixel-level Non-local Method for Real-world Image Denoising](https://arxiv.org/pdf/1906.06834.pdf)|-|5|
|TIP|[Image Denoising via Sequential Ensemble Learning](https://cpb-us-w2.wpmucdn.com/blog.nus.edu.sg/dist/8/10877/files/2020/03/TIP2020_ensemble.pdf)|-|2|
|TIP|[Connecting Image Denoising and High-Level Vision Tasks via Deep Learning](https://arxiv.org/pdf/1809.01826.pdf)|-|16|
|CVPR|[Memory-Efficient Hierarchical Neural Architecture Search for Image Denoising](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Memory-Efficient_Hierarchical_Neural_Architecture_Search_for_Image_Denoising_CVPR_2020_paper.pdf)|-|3|
|CVPR|[A Physics-based Noise Formation Model for Extreme Low-light Raw Denoising](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_A_Physics-Based_Noise_Formation_Model_for_Extreme_Low-Light_Raw_Denoising_CVPR_2020_paper.pdf)|[Pytorch](https://github.com/Vandermode/ELD)|1|
|CVPR|[Supervised Raw Video Denoising With a Benchmark Dataset on Dynamic Scenes](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yue_Supervised_Raw_Video_Denoising_With_a_Benchmark_Dataset_on_Dynamic_CVPR_2020_paper.pdf)|Both|Video|Real|[Pytorch](https://github.com/cao-cong/RViDeNet)|1|
|CVPR|[Transfer Learning From Synthetic to Real-Noise Denoising With Adaptive Instance Normalization](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kim_Transfer_Learning_From_Synthetic_to_Real-Noise_Denoising_With_Adaptive_Instance_CVPR_2020_paper.pdf)|-|1|
|CVPR|[Self2Self With Dropout: Learning Self-Supervised Denoising From Single Image](https://openaccess.thecvf.com/content_CVPR_2020/papers/Quan_Self2Self_With_Dropout_Learning_Self-Supervised_Denoising_From_Single_Image_CVPR_2020_paper.pdf)|-|3|
|CVPR|[Noisier2Noise: Learning to Denoise From Unpaired Noisy Data](https://openaccess.thecvf.com/content_CVPR_2020/papers/Moran_Noisier2Noise_Learning_to_Denoise_From_Unpaired_Noisy_Data_CVPR_2020_paper.pdf)|-|3|
|CVPR|[Joint Demosaicing and Denoising With Self Guidance](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Joint_Demosaicing_and_Denoising_With_Self_Guidance_CVPR_2020_paper.pdf)|-|2|
|CVPR|[FastDVDnet: Towards Real-Time Deep Video Denoising Without Flow Estimation](https://openaccess.thecvf.com/content_CVPR_2020/papers/Tassano_FastDVDnet_Towards_Real-Time_Deep_Video_Denoising_Without_Flow_Estimation_CVPR_2020_paper.pdf)|RGB|Video|AWGN|-|1|
|CVPR|[CycleISP: Real Image Restoration via Improved Data Synthesis](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zamir_CycleISP_Real_Image_Restoration_via_Improved_Data_Synthesis_CVPR_2020_paper.pdf)|[Pytorch](https://github.com/swz30/CycleISP)|4|
|CVPR|[Basis Prediction Networks for Effective Burst Denoising With Large Kernels](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xia_Basis_Prediction_Networks_for_Effective_Burst_Denoising_With_Large_Kernels_CVPR_2020_paper.pdf)|-|2|
|CVPR|[Superkernel Neural Architecture Search for Image Denoising](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w31/Mozejko_Superkernel_Neural_Architecture_Search_for_Image_Denoising_CVPRW_2020_paper.pdf)|-|2|
|ECCV|[Spatial-Adaptive Network for Single Image Denoising](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123750171.pdf)|-|0|
|ECCV|[A Decoupled Learning Scheme for Real-world Burst Denoising from Raw Images](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700154.pdf)|-|0|
|ECCV|[Burst Denoising via Temporally Shifted Wavelet Transforms](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580239.pdf)|-|0|
|ECCV|[Unpaired Learning of Deep Image Denoising](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490341.pdf)|[Pytorch](https://github.com/XHWXD/DBSN)|0|
|ECCV|[Dual Adversarial Network: Toward Real-world Noise Removal and Noise Generation](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123550035.pdf)|[Pytorch](https://github.com/zsyOAOA/DANet)|1|
|ECCV|[Learning Camera-Aware Noise Models](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690341.pdf)|[Pytorch](https://github.com/arcchang1236/CA-NoiseGAN)|0|
|ECCV|[Practical Deep Raw Image Denoising on Mobile Devices](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123510001.pdf)|Raw|Single|PG|-|0|
|ECCV|[Reconstructing the Noise Manifold for Image Denoising](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123540596.pdf)|-|0|
|NN|[Deep Learning on Image Denoising : An Overview](https://arxiv.org/pdf/1912.13171.pdf)|-|19|
|WACV|[Identifying recurring patterns with deep neural networks for natural image denoising](http://openaccess.thecvf.com/content_WACV_2020/papers/Xia_Identifying_Recurring_Patterns_with_Deep_Neural_Networks_for_Natural_Image_WACV_2020_paper.pdf)|-|3|
|ICASSP|[Attention Mechanism Enhanced Kernel Prediction Networks for Denoising of Burst Images](https://arxiv.org/pdf/1910.08313.pdf)|[Pytorch](https://github.com/z-bingo/Attention-Mechanism-Enhanced-KPN)|1|
|Arxiv|[Low-light Image Restoration with Short- and Long-exposure Raw Pairs](https://arxiv.org/pdf/2007.00199.pdf)|-|0|

## 2019  

|Pub|Title|Color|Image|NoiseCode|Cite|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|TIP|[Optimal combination of image denoisers](https://arxiv.org/pdf/1711.06712.pdf)|-|8|
|TIP|[High ISO JPEG Image Denoising by Deep Fusion of Collaborative and Convolutional Filtering](https://sci-hub.se/https://ieeexplore.ieee.org/abstract/document/8684332/)|-|4|
|TIP|[Texture variation adaptive image denoising with nonlocal PCA](https://arxiv.org/pdf/1810.11282.pdf)|-|2|
|TIP|[Color Image and Multispectral Image Denoising Using Block Diagonal Representation](https://arxiv.org/pdf/1902.03954.pdf)|-|7|
|TIP|Tchebichef and Adaptive Steerable-Based Total Variation Model for Image Denoising|-|12|
|TIP|[Iterative Joint Image Demosaicking and Denoising Using a Residual Denoising Network](https://arxiv.org/pdf/1807.06403.pdf)|-|21|
|TIP|Content-Adaptive Noise Estimation for Color Images with Cross-Channel Noise Modeling|-|0|
|TPAMI|[Real-world Image Denoising with Deep Boosting](https://sci-hub.se/https://ieeexplore.ieee.org/abstract/document/8733117/)|[Tensorflow](https://github.com/ngchc/deepBoosting)|9|
|JVCIR|Vst-net: Variance-stabilizing transformation inspired network for poisson denoising|[Matlab](https://github.com/yqx7150/VST-Net)|3|
|NIPS|[Variational Denoising Network: Toward Blind Noise Modeling and Removal](https://papers.nips.cc/paper/8446-variational-denoising-network-toward-blind-noise-modeling-and-removal.pdf)|-|20|
|NIPS|[High-Quality Self-Supervised Deep Image Denoising](http://papers.nips.cc/paper/8920-high-quality-self-supervised-deep-image-denoising.pdf)|-|22|
|ICML|[Noise2Self: Blind Denoising by Self-Supervision](https://arxiv.org/pdf/1901.11365.pdf)|[Pytorch](https://github.com/czbiohub/noise2self)|71|
|ICML|[Plug-and-play methods provably converge with properly trained denoisers](https://arxiv.org/pdf/1905.05406.pdf)|-|34|
|CVPR|[Unsupervised Domain Adaptation for ToF Data Denoising with Adversarial Learning](https://openaccess.thecvf.com/content_CVPR_2019/papers/Agresti_Unsupervised_Domain_Adaptation_for_ToF_Data_Denoising_With_Adversarial_Learning_CVPR_2019_paper.pdf)|-|7|
|CVPR|[Robust Subspace Clustering with Independent and Piecewise Identically Distributed Noise Modeling](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Robust_Subspace_Clustering_With_Independent_and_Piecewise_Identically_Distributed_Noise_CVPR_2019_paper.pdf)|-|2|
|CVPR|[Toward convolutional blind denoising of real photographs](http://openaccess.thecvf.com/content_CVPR_2019/papers/Guo_Toward_Convolutional_Blind_Denoising_of_Real_Photographs_CVPR_2019_paper.pdf)|[Matlab](https://github.com/GuoShi28/CBDNet)|139|
|CVPR|[FOCNet: A Fractional Optimal Control Network for Image Denoising](http://openaccess.thecvf.com/content_CVPR_2019/papers/Jia_FOCNet_A_Fractional_Optimal_Control_Network_for_Image_Denoising_CVPR_2019_paper.pdf)|-|22|
|CVPR|[Noise2void-learning denoising from single noisy images](http://openaccess.thecvf.com/content_CVPR_2019/papers/Krull_Noise2Void_-_Learning_Denoising_From_Single_Noisy_Images_CVPR_2019_paper.pdf)|-|101|
|CVPR|[Unprocessing images for learned raw denoising](http://openaccess.thecvf.com/content_CVPR_2019/papers/Brooks_Unprocessing_Images_for_Learned_Raw_Denoising_CVPR_2019_paper.pdf)|-|75|
|CVPR|[Training deep learning based image denoisers from undersampled measurements without ground truth and without image prior](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhussip_Training_Deep_Learning_Based_Image_Denoisers_From_Undersampled_Measurements_Without_CVPR_2019_paper.pdf)|-|12|
|CVPR|[Model-blind video denoising via frame-to-frame training](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ehret_Model-Blind_Video_Denoising_via_Frame-To-Frame_Training_CVPR_2019_paper.pdf)|[other](https://github.com/tehret/blind-denoising)|16|
|ICCV|[Self-Guided Network for Fast Image Denoising](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gu_Self-Guided_Network_for_Fast_Image_Denoising_ICCV_2019_paper.pdf)|-|17|
|ICCV|[Noise flow: Noise modeling with conditional normalizing flows](https://openaccess.thecvf.com/content_ICCV_2019/papers/Abdelhamed_Noise_Flow_Noise_Modeling_With_Conditional_Normalizing_Flows_ICCV_2019_paper.pdf)|-|19|
|ICCV|[Joint Demosaicking and Denoising by Fine-Tuning of Bursts of Raw Images](https://openaccess.thecvf.com/content_ICCV_2019/papers/Ehret_Joint_Demosaicking_and_Denoising_by_Fine-Tuning_of_Bursts_of_Raw_ICCV_2019_paper.pdf)|-|4|
|ICCV|[Fully Convolutional Pixel Adaptive Image Denoiser](https://openaccess.thecvf.com/content_ICCV_2019/papers/Cha_Fully_Convolutional_Pixel_Adaptive_Image_Denoiser_ICCV_2019_paper.pdf)|[Keras](https://github.com/csm9493/FC-AIDE-Keras)|8|
|ICCV|[Enhancing Low Light Videos by Exploring High Sensitivity Camera Noise](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Enhancing_Low_Light_Videos_by_Exploring_High_Sensitivity_Camera_Noise_ICCV_2019_paper.pdf)|-|4|
|ICCV|[CIIDefence: Defeating Adversarial Attacks by Fusing Class-Specific Image Inpainting and Image Denoising](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gupta_CIIDefence_Defeating_Adversarial_Attacks_by_Fusing_Class-Specific_Image_Inpainting_and_ICCV_2019_paper.pdf)|-|6|
|ICCV|[Real Image Denoising with Feature Attention](https://arxiv.org/pdf/1904.07396.pdf)|-|46|
|CVPRW|[GRDN:Grouped Residual Dense Network for Real Image Denoising and GAN-based Real-world Noise Modeling](http://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Kim_GRDNGrouped_Residual_Dense_Network_for_Real_Image_Denoising_and_GAN-Based_CVPRW_2019_paper.pdf)|-|25|
|CVPRW|[Learning raw image denoising with bayer pattern unification and bayer preserving augmentation](https://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Liu_Learning_Raw_Image_Denoising_With_Bayer_Pattern_Unification_and_Bayer_CVPRW_2019_paper.pdf)|-|15|
|CVPRW|[Deep iterative down-up CNN for image denoising](http://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Yu_Deep_Iterative_Down-Up_CNN_for_Image_Denoising_CVPRW_2019_paper.pdf)|-|18|
|CVPRW|[Densely Connected Hierarchical Network for Image Denoising](https://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Park_Densely_Connected_Hierarchical_Network_for_Image_Denoising_CVPRW_2019_paper.pdf)|-|16|
|CVPRW|[ViDeNN: Deep Blind Video Denoising](http://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Claus_ViDeNN_Deep_Blind_Video_Denoising_CVPRW_2019_paper.pdf)|-|10|
|CVPRW|[Real Photographs Denoising With Noise Domain Adaptation and Attentive Generative Adversarial Network](http://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Lin_Real_Photographs_Denoising_With_Noise_Domain_Adaptation_and_Attentive_Generative_CVPRW_2019_paper.pdf)|-|6|
|CVPRW|[Learning Deep Image Priors for Blind Image Denoising](http://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Hou_Learning_Deep_Image_Priors_for_Blind_Image_Denoising_CVPRW_2019_paper.pdf)|-|2|
|ICIP|[DVDnet: A fast network for deep video denoising](https://arxiv.org/pdf/1906.11890.pdf)|RGB|Video|AWGN|[Pytorch](https://github.com/m-tassano/dvdnet)|5|
|ICIP|[Multi-kernel prediction networks for denoising of burst images](https://arxiv.org/pdf/1902.05392.pdf)|-|4|
|ICIP|A non-local cnn for video denoising|-|9|
|AAAI|Adaptation Strategies for Applying AWGN-based Denoiser to Realistic Noise|-|3|
|arxiv|[When AWGN-based Denoiser Meets Real Noises](https://arxiv.org/pdf/1904.03485.pdf)|[Pytorch](https://github.com/yzhouas/PD-Denoising-pytorch)|10|
|arxiv|[Generating training data for denoising real rgb images via camera pipeline simulation](https://arxiv.org/pdf/1904.08825.pdf)|-|7|
|arxiv|[Learning Deformable Kernels for Image and Video Denoising](https://arxiv.org/pdf/1904.06903.pdf)|-|6|
|arxiv|[Gan2gan: Generative noise learning for blind image denoising with single noisy images](https://arxiv.org/pdf/1905.10488.pdf)|-|5|

## 2018  

|Pub|Title|Code|Cite|
|:---:|:---:|:---:|:---:|
|TIP|Weighted Tensor Rank-1 Decomposition for Nonlocal Image Denoising|-|6|
|TIP|Towards Optimal Denoising of Image Contrast|-|４|
|TIP|[Time-of-Flight Range Measurement in Low- sensing Environment : Noise Analysis and Complex-domain Non-local Denoising](https://www.researchgate.net/profile/Mihail_Georgiev4/publication/323233188_Time-of-Flight_Range_Measurement_in_Low-Sensing_Environment_Noise_Analysis_and_Complex-Domain_Non-Local_Denoising/links/5b2373750f7e9b0e374893a7/Time-of-Flight-Range-Measurement-in-Low-Sensing-Environment-Noise-Analysis-and-Complex-Domain-Non-Local-Denoising.pdf)|-|4|
|TIP|[Statistical Nearest Neighbors for Image Denoising](https://research.nvidia.com/sites/default/files/pubs/2018-09_Statistical-Nearest-Neighbors/Statistical%20Nearest%20Neighbors%20for%20Image%20Denoising.pdf)|-|10|
|TIP|[Joint Denoising / Compression of Image Contours via Shape Prior and Context Tree](https://arxiv.org/pdf/1705.00268.pdf)|-|5|
|TIP|[Image Restoration by Iterative Denoising and Backward Projections](https://arxiv.org/pdf/1710.06647.pdf)|-|45|
|TIP|Corrupted reference image quality assessment of denoised images|-|3|
|TIP|[FFDNet: Toward a fast and flexible solution for CNN-based image denoising](https://arxiv.org/pdf/1710.04026.pdf)|[Matlab](https://github.com/cszn/FFDNet)|437|
|TIP|[External prior guided internal prior learning for real-world noisy image denoising](https://arxiv.org/pdf/1705.04505.pdf)|-|60|
|TIP|[Class-aware fully convolutional Gaussian and Poisson denoising](https://arxiv.org/pdf/1808.06562.pdf)|[Tensorflow](https://github.com/TalRemez/deep_class_aware_denoising)|27|
|TIP|[VIDOSAT: High-dimensional sparsifying transform learning for online video denoising](https://arxiv.org/pdf/1710.00947.pdf)|-|17|
|TIP|[Effective and fast estimation for image sensor noise via constrained weighted least squares](https://www.researchgate.net/profile/Jiantao_Zhou/publication/323563338_Effective_and_Fast_Estimation_for_Image_Sensor_Noise_Via_Constrained_Weighted_Least_Squares/links/5acdcaa6a6fdcc87840afac1/Effective-and-Fast-Estimation-for-Image-Sensor-Noise-Via-Constrained-Weighted-Least-Squares.pdf)|-|11|
|ToG|Denoising with kernel prediction and asymmetric loss functions|-|46|
|TMM|Gradient prior-aided cnn denoiser with separable convolution-based optimization of feature dimension|-|9|
|NIPS|[Training deep learning based denoisers without ground truth data](https://papers.nips.cc/paper/7587-training-deep-learning-based-denoisers-without-ground-truth-data.pdf)|-|31|
|ICML|[Noise2Noise: Learning Image Restoration without Clean Data](https://arxiv.org/pdf/1803.04189.pdf)|-|280|
|CVPR|[Burst denoising with kernel prediction networks](http://openaccess.thecvf.com/content_cvpr_2018/papers/Mildenhall_Burst_Denoising_With_CVPR_2018_paper.pdf)|-|94|
|CVPR|[Image Blind Denoising With Generative Adversarial Network Based Noise Modeling](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Image_Blind_Denoising_CVPR_2018_paper.pdf)|-|148|
|CVPR|[Universal Denoising Networks : A Novel CNN Architecture for Image Denoising](http://openaccess.thecvf.com/content_cvpr_2018/papers/Lefkimmiatis_Universal_Denoising_Networks_CVPR_2018_paper.pdf)|[Matlab](https://github.com/cig-skoltech/UDNet)|127|
|ECCV|[Deep burst denoising](http://openaccess.thecvf.com/content_ECCV_2018/papers/Clement_Godard_Deep_Burst_Denoising_ECCV_2018_paper.pdf)|-|41|
|ECCV|[Deep boosting for image denoising](http://openaccess.thecvf.com/content_ECCV_2018/papers/Chang_Chen_Deep_Boosting_for_ECCV_2018_paper.pdf)|-|27|
|ECCV|[A trilateral weighted sparse coding scheme for real-world image denoising](http://openaccess.thecvf.com/content_ECCV_2018/papers/XU_JUN_A_Trilateral_Weighted_ECCV_2018_paper.pdf)|-|88|
|ECCV|[Deep image demosaicking using a cascade of convolutional residual denoising networks](https://openaccess.thecvf.com/content_ECCV_2018/papers/Filippos_Kokkinos_Deep_Image_Demosaicking_ECCV_2018_paper.pdf)|-|33|
|IJCAI|[Connecting image denoising and high-level vision tasks via deep learning](https://arxiv.org/pdf/1809.01826.pdf)|-|16|
|IJCAI|[When image denoising meets high-level vision tasks: A deep learning approach](https://arxiv.org/pdf/1706.04284.pdf)|-|97|
|JVCIR|[RENOIR–A dataset for real low-light image noise reduction](https://arxiv.org/pdf/1409.8230.pdf)|-|57|
|TCI|[Convolutional neural networks for noniterative reconstruction of compressively sensed images](https://arxiv.org/pdf/1708.04669.pdf)|-|45|
|ACCV|[Dn-resnet: Efficient deep residual network for image denoising](https://arxiv.org/pdf/1810.06766.pdf)|-|10|
|ICIP|[Image Denoising for Image Retrieval by Cascading a Deep Quality Assessment Network](http://www.ee.iisc.ac.in/new/people/faculty/soma.biswas/Papers/biju_icip2018.pdf)|-|3|
|arxiv|[Correction by projection: Denoising images with generative adversarial networks](https://arxiv.org/pdf/1803.04477.pdf)|-|29|
|arxiv|[Non-local video denoising by CNN](https://arxiv.org/pdf/1811.12758.pdf)|[Pytorch](https://github.com/axeldavy/vnlnet)|15|
|arxiv|[Iterative residual network for deep joint image demosaicking and denoising](https://arxiv.org/pdf/1807.06403.pdf)|-|9|
|arxiv|[Fully convolutional pixel adaptive image denoiser](https://arxiv.org/pdf/1807.07569.pdf)|-|8|
|arxiv|[Fast, trainable, multiscale denoising](https://arxiv.org/pdf/1802.06130.pdf)|-|4|
|arxiv|[Deep learning for image denoising: a survey](https://arxiv.org/pdf/1810.05052.pdf)|-|40|

## 2017  

|Publ|Title|Code|Cite|
|:---:|:---:|:---:|:---:|
|TIP|[Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising](https://arxiv.org/pdf/1608.03981.pdf)|-|2250|
|TIP|Improved Denoising via Poisson Mixture Modeling of Image Sensor Noise|-|14|
|TIP|Reweighted Low-Rank Matrix Analysis with Structural Smoothness for Image Denoising|-|22|
|TIP|Category-specific object image denoising|-|21|
|TIP|[Affine Non-Local Means Image Denoising](https://repositori.upf.edu/bitstream/handle/10230/37095/ballester_trans26_affi.pdf?sequence=1&isAllowed=y)|-|24|
|CVPR|[Image Denoising via CNNs: An Adversarial Approach](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Divakar_Image_Denoising_via_CVPR_2017_paper.pdf)|-|43|
|CVPR|[Non-local color image denoising with convolutional neural networks](http://openaccess.thecvf.com/content_cvpr_2017/papers/Lefkimmiatis_Non-Local_Color_Image_CVPR_2017_paper.pdf)|-|162|
|CVPR|[Learning Deep CNN Denoiser Prior for Image Restoration](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhang_Learning_Deep_CNN_CVPR_2017_paper.pdf)|-|669|
|ICCV|[Learning Proximal Operators : Using Denoising Networks for Regularizing Inverse Imaging Problems](https://openaccess.thecvf.com/content_ICCV_2017/papers/Meinhardt_Learning_Proximal_Operators_ICCV_2017_paper.pdf)|-|130|
|ICCV|[Multi-channel Weighted Nuclear Norm Minimization for Real Color Image Denoising](https://openaccess.thecvf.com/content_ICCV_2017/papers/Xu_Multi-Channel_Weighted_Nuclear_ICCV_2017_paper.pdf)|-|134|
|ICCV|[Joint Adaptive Sparsity and Low-Rankness on the Fly: An Online Tensor Reconstruction Scheme for Video Denoising](https://openaccess.thecvf.com/content_ICCV_2017/papers/Wen_Joint_Adaptive_Sparsity_ICCV_2017_paper.pdf)|-|29|
|ICCV|[Blob Reconstruction Using Unilateral Second Order Gaussian Kernels with Application to High-ISO Long-Exposure Image Denoising](https://openaccess.thecvf.com/content_ICCV_2017/papers/Wang_Blob_Reconstruction_Using_ICCV_2017_paper.pdf)|-|8|
|ICIP|[Image denoising using group sparsity residual and external nonlocal self-similarity prior](https://arxiv.org/pdf/1701.00723.pdf)|-|6|
|arxiv|[Block-matching convolutional neural network for image denoising](https://arxiv.org/pdf/1704.00524.pdf)|-|36|
|arxiv|[Learning pixel-distribution prior with wider convolution for image denoising](https://arxiv.org/pdf/1707.09135.pdf)|[Matlab](https://github.com/cswin/WIN)|10|
|arxiv|[Chaining identity mapping modules for image denoising](https://arxiv.org/pdf/1712.02933.pdf)|-|8|
|ICTAI|[Dilated deep residual network for image denoising](https://arxiv.org/pdf/1708.05473.pdf)|-|34|

## before 2017  

|Year|Publication|Title|Code|Citation|
|:---:|:---:|:---:|:---:|:---:|
|2016|CVPR|[Deep Gaussian conditional random field network: A model-based deep network for discriminative denoising](https://openaccess.thecvf.com/content_cvpr_2016/papers/Vemulapalli_Deep_Gaussian_Conditional_CVPR_2016_paper.pdf)|-|53|
|2016|CVPR|[From Noise Modeling to Blind Image Denoising](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhu_From_Noise_Modeling_CVPR_2016_paper.pdf)|-|44|
|2016|TIP|Patch-based video denoising with optical flow estimation|-|64|
|2016|ToG|Deep joint demosaicking and denoising|-|205|
|2016|ICASSP|Fast depth image denoising and enhancement using a deep convolutional network|-|41|
|2015|ICCV|[An efficient statistical method for image noise level estimation](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Chen_An_Efficient_Statistical_ICCV_2015_paper.pdf)|-|103|
|2015|TIP|Image-specific prior adaptation for denoising|-|16|
|2015|IPOL|[The noise clinic: a blind image denoising algorithm](http://www.ipol.im/pub/art/2015/125/article_lr.pdf)|-|76|
|2014|TIP|Practical signal-dependent noise parameter estimation from a single noisy image|-|55|
|2014|-|[Photon, Poisson Noise](http://people.csail.mit.edu/hasinoff/pubs/hasinoff-photon-2011-preprint.pdf)|-|67|
|2012|CVPR|[Image denoising: Can plain neural networks compete with BM3D?](https://hcburger.com/files/neuraldenoising.pdf)|-|911|
|2012|ICIP|The dominance of Poisson noise in color digital cameras|-|22|
|2009|SP|[Clipped noisy images: Heteroskedastic modeling and practical denoising](https://www.researchgate.net/profile/Alessandro_Foi/publication/220227880_Clipped_noisy_images_Heteroskedastic_modeling_and_practical_denoising/links/5b7d594c299bf1d5a71c4b11/Clipped-noisy-images-Heteroskedastic-modeling-and-practical-denoising.pdf)|-|113|
|2008|TIP|[Practical Poissonian-Gaussian noise modeling and fitting for single-image raw-data](https://core.ac.uk/download/pdf/194121585.pdf)|[Matlab]()|552|
|2007|TIP|[Image denoising by sparse 3-D transform-domain collaborative filtering](http://web.eecs.utk.edu/~hqi/ece692/references/noise-BM3D-tip07.pdf)|-|6029|
|2007|TPAMI|[Automatic estimation and removal of noise from a single image](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.228.3525&rep=rep1&type=pdf)|-|520|
|2005|CVPR|[A non-local algorithm for image denoising](http://audio.rightmark.org/lukin/msu/NonLocal.pdf)|-|6020|
|2019|Books|CMOS: Circuit Design, Layout, and Simulation: Forth Edition|-|4870|
|2018|Books|Denoising of photographic images and video: fundamentals, open challenges and new trends|-|5|
