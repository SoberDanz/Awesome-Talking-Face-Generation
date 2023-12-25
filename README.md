# Awesome-Talking-Face-Generation

This is a repository for organizing papers and codes about Talking-Face-Generation(TFG) for computer vision.

Besides,the commonly-used datasets and metrics for TFG are also introduced.



💫 This project is constantly being updated,any suggestions are welcomed!

##  Papers

### 2023

| Title                                                        |   Venue    |                     Dataset                      |                             PDF                              |                         CODE                          |
| :----------------------------------------------------------- | :--------: | :----------------------------------------------: | :----------------------------------------------------------: | :---------------------------------------------------: |
| DreamTalk: When Expressive Talking Head Generation Meets Diffusion Probabilistic Models | arXiv 2023 |             MEAD & HDTF &  Voxceleb2             |           [PDF](https://arxiv.org/abs/2312.09767)            |                           -                           |
| GMTalker: Gaussian Mixture based Emotional talking video Portraits | arXiv 2023 |                    MEAD & LSP                    |           [PDF](https://arxiv.org/abs/2312.07669)            |                           -                           |
| DiT-Head: High-Resolution Talking Head Synthesis using Diffusion Transformers | arXiv 2023 |                       HDTF                       |           [PDF](https://arxiv.org/abs/2312.06400)            |                           -                           |
| R2-Talker: Realistic Real-Time Talking Head Synthesis with Hash Grid Landmarks Encoding and Progressive Multilayer Conditioning | arXiv 2023 |                        -                         |           [PDF](https://arxiv.org/abs/2312.05572)            |                           -                           |
| FT2TF: First-Person Statement Text-To-Talking Face Generation | arXiv 2023 |                   LRS2 & LRS3                    |           [PDF](https://arxiv.org/abs/2312.05430)            |                           -                           |
| VividTalk: One-Shot Audio-Driven Talking Head Generation Based on 3D Hybrid Prior | arXiv 2023 |                 HDTF &  VoxCeleb                 |           [PDF](https://arxiv.org/abs/2312.01841)            |                           -                           |
| SyncTalk: The Devil is in the Synchronization for Talking Head Synthesi | arXiv 2023 |                       LRS3                       |           [PDF](https://arxiv.org/abs/2311.17590)            |                           -                           |
| GeneFace: Generalized and High-Fidelity Audio-Driven 3D Talking Face Synthesis | ICLR 2023  |                       LRS3                       |                           [PDF]()                            |      [CODE](https://github.com/yerfor/GeneFace)       |
| GAIA: Zero-shot Talking Avatar Generation                    | arXiv 2023 |           dataset from diverse sources           |           [PDF](https://arxiv.org/abs/2311.15230)            |                           -                           |
| Efficient Region-Aware Neural Radiance Fields for High-Fidelity Talking Portrait Synthesis | ICCV 2023  |                        -                         | [PDF](https://openaccess.thecvf.com/content/ICCV2023/html/Li_Efficient_Region-Aware_Neural_Radiance_Fields_for_High-Fidelity_Talking_Portrait_Synthesis_ICCV_2023_paper.html) |    [CODE](https://github.com/Fictionarry/ER-NeRF)     |
| Implicit Identity Representation Conditioned Memory Compensation Network for Talking Head Video Generation | ICCV 2023  |               VoxCeleb1 &  CelebV                |           [PDF](https://arxiv.org/abs/2307.09906)            | [CODE](https://github.com/harlanhong/ICCV2023-MCNET)  |
| MODA: Mapping-Once Audio-driven Portrait Animation with Dual Attentions | ICCV 2023  |                    HDTF & LSP                    |           [PDF](https://arxiv.org/abs/2307.10008)            |                           -                           |
| Efficient Emotional Adaptation for Audio-Driven Talking-Head Generation | ICCV 2023  |            Celeb2 & MEAD & LRW & MEAD            |           [PDF](https://arxiv.org/abs/2309.04946)            |      [CODE](https://github.com/yuangan/eat_code)      |
| EMMN: Emotional Motion Memory Network for Audio-driven Emotional Talking Face Generation | ICCV 2023  |                    MEAD & LRW                    | [PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Tan_EMMN_Emotional_Motion_Memory_Network_for_Audio-driven_Emotional_Talking_Face_ICCV_2023_paper.pdf) |                           -                           |
| Emotional Listener Portrait: Realistic Listener Motion Simulation in Conversation | ICCV 2023  | ViCo and the dataset proposed by Learning2Listen | [PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Song_Emotional_Listener_Portrait_Neural_Listener_Head_Generation_with_Emotion_ICCV_2023_paper.pdf) |                           -                           |
| MetaPortrait: Identity-Preserving Talking Head Generation with Fast Personalized Adaptation | CVPR 2023  |                VoxCeleb2 &  HDTF                 |           [PDF](https://arxiv.org/abs/2212.08062)            | [CODE](https://github.com/Meta-Portrait/MetaPortrait) |
| Implicit Neural Head Synthesis via Controllable Local Deformation Fields | CVPR 2023  |                        -                         |           [PDF](https://arxiv.org/abs/2304.11113)            |                           -                           |
| LipFormer: High-fidelity and Generalizable Talking Face Generation with A Pre-learned Facial Codebook | CVPR 2023  |                   LRS2 & FFHQ                    | [PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_LipFormer_High-Fidelity_and_Generalizable_Talking_Face_Generation_With_a_Pre-Learned_CVPR_2023_paper.pdf) |                           -                           |
| GANHead: Towards Generative Animatable Neural Head Avatars   | CVPR 2023  |                FaceVerse-Dataset                 |          [PDF](https://arxiv.org/abs/2304.03950v1)           |      [CODE](https://github.com/wsj-sjtu/GANHead)      |
| Parametric Implicit Face Representation for Audio-Driven Facial Reenactment | CVPR 2023  |           HDTF & Testset1  & Testset 2           | [PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Parametric_Implicit_Face_Representation_for_Audio-Driven_Facial_Reenactment_CVPR_2023_paper.pdf) |                           -                           |
| Identity-Preserving Talking Face Generation with Landmark and Appearance Priors | CVPR 2023  |                   LRS2 & LRS3                    |           [PDF](https://arxiv.org/abs/2305.08293)            |    [CODE](https://github.com/Weizhi-Zhong/IP_LAP)     |
| StyleSync: High-Fidelity Generalized and Personalized Lip Sync in Style-based Generator | CVPR 2023  |                  LRW & VoxCeleb                  |         [PDF](https://arxiv.org/pdf/2305.05445.pdf)          |                           -                           |
| High-fidelity Generalized Emotional Talking Face Generation with Multi-modal Emotion Space Learning | CVPR 2023  |                       MEAD                       |           [PDF](https://arxiv.org/abs/2305.02572)            |                           -                           |
| Seeing What You Said: Talking Face Generation Guided by a Lip Reading Expert | CVPR 2023  |                    LRS2 & LRW                    |           [PDF](https://arxiv.org/abs/2303.17480)            |      [CODE](https://github.com/Sxjdwang/TalkLip)      |
| OTAvatar : One-shot Talking Face Avatar with Controllable Tri-plane Rendering | CVPR 2023  |                HDTF & e Multiface                |           [PDF](https://arxiv.org/abs/2303.14662)            |     [CODE](https://github.com/theEricMa/OTAvatar)     |
| Style Transfer for 2D Talking Head Animation                 | arXiv 2023 |                    VoxCeleb2                     |           [PDF](https://arxiv.org/abs/2303.09799)            |                           -                           |
| StyleTalk: One-shot Talking Head Generation with Controllable Speaking Styles | AAAI 2023  |                   MEAD & HDTF                    |           [PDF](https://arxiv.org/abs/2301.01081)            | [CODE](https://github.com/FuxiVirtualHuman/styletalk) |

### 2022

| Title                                                        |    Venue    |                  Dataset                   |                             PDF                              |                         CODE                         |
| :----------------------------------------------------------- | :---------: | :----------------------------------------: | :----------------------------------------------------------: | :--------------------------------------------------: |
| SyncTalkFace: Talking Face Generation with Precise Lip-syncing via Audio-Lip Memory |  AAAI 2022  |           LRW & LRS2 & BBC News            | [PDF](https://ojs.aaai.org/index.php/AAAI/article/download/20102/19861) |                          -                           |
| Progressive Disentangled Representation Learning for Fine-Grained Controllable Talking Head Synthesis |  CVPR 2022  |              VoxCeleb2 & Mead              |           [PDF](https://arxiv.org/abs/2211.14506)            |                          -                           |
| Compressing Video Calls using Synthetic Talking Heads        |  BMVC 2022  |                     -                      |           [PDF](https://arxiv.org/abs/2210.03692)            |                          -                           |
| Synthesizing Photorealistic Virtual Humans Through Cross-modal Disentanglement | arXiv 2022  |                     -                      |         [PDF](https://arxiv.org/pdf/2209.01320.pdf)          |                          -                           |
| StyleTalker: One-shot Style-based Audio-driven Talking Head Video Generation | arXiv 2022  |                 Voxceleb2                  |         [PDF](https://arxiv.org/pdf/2208.10922.pdf)          |                          -                           |
| Talking Head from Speech Audio using a Pre-trained Image Generato | ACM MM 2022 |             TCD-TIMIT &  GRID              |         [PDF](https://arxiv.org/pdf/2209.04252.pdf)          |                          -                           |
| Learning Dynamic Facial Radiance Fields for Few-Shot Talking Head Synthesis |  ECCV 2022  |                     -                      |         [PDF](https://arxiv.org/pdf/2207.11770.pdf)          |        [CODE](https://github.com/sstzal/DFRF)        |
| Semantic-Aware Implicit Neural Audio-Driven Video Portrait Generation |  ECCV 2022  |                     -                      |         [PDF](https://arxiv.org/pdf/2201.07786.pdf)          |    [CODE](https://github.com/alvinliu0/SSP-NeRF)     |
| Text2Video: Text-driven Talking-head Video Synthesis with Phonetic Dictionary | ICASSP 2022 |                  VidTIMIT                  |         [PDF](https://arxiv.org/pdf/2104.14631.pdf)          |   [CODE](https://github.com/sibozhang/Text2Video)    |
| Emotion-Controllable Generalized Talking Face Generation     | IJCAI 2022  |          MEAD & CREMA-D & RAVDESS          |    [PDF](https://www.ijcai.org/proceedings/2022/0184.pdf)    |                          -                           |
| Show Me What and Tell Me How: Video Synthesis via Multimodal Conditioning |  CVPR 2022  | Shapes & MUG & iPER &  Multimodal VoxCeleb |         [PDF](https://arxiv.org/pdf/2203.02573.pdf)          |    [CODE](https://github.com/snap-research/MMVID)    |
| Depth-Aware Generative Adversarial Network for Talking Head Video Generation |  CVPR 2022  |             VoxCeleb1 & CelebV             |         [PDF](https://arxiv.org/pdf/2203.06605.pdf)          | [CODE](https://github.com/harlanhong/CVPR2022-DaGAN) |
| Expressive Talking Head Generation with Granular Audio-Visual Control |  CVPR 2022  |              Voxceleb2 & MEAD              | [PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Liang_Expressive_Talking_Head_Generation_With_Granular_Audio-Visual_Control_CVPR_2022_paper.pdf) |                          -                           |


### 2021

| Title                                                        |   Venue    |           Dataset           |                             PDF                              |                             CODE                             |
| :----------------------------------------------------------- | :--------: | :-------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Audio-Driven Emotional Video Portraits                       | CVPR 2021  |         MEAD & LRW          |         [PDF](https://arxiv.org/pdf/2104.07452.pdf)          |           [CODE](https://github.com/jixinya/EVP/)            |
| Pose-Controllable Talking Face Generation by Implicitly Modularized Audio-Visual Representation | CVPR 2021  |       VoxCeleb2& LRW        | [PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhou_Pose-Controllable_Talking_Face_Generation_by_Implicitly_Modularized_Audio-Visual_Representation_CVPR_2021_paper.pdf) | [CODE](https://github.com/Hangz-nju-cuhk/Talking-Face_PC-AVS) |
| Flow-guided One-shot Talking Face Generation with a High-resolution Audio-visual Dataset | CVPR 2021  |            HDTF             | [PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Flow-Guided_One-Shot_Talking_Face_Generation_With_a_High-Resolution_Audio-Visual_Dataset_CVPR_2021_paper.pdf) |            [CODE](https://github.com/MRzzm/HDTF)             |
| Write-a-speaker: Text-based Emotional and Rhythmic Talking-head Generation | AAAI 2021  |        Mocap dataset        |         [PDF](https://arxiv.org/pdf/2104.07995.pdf)          |                              -                               |
| Audio2Head: Audio-driven One-shot Talking-head Generation with Natural Head Motion | IJCAI 2021 |    VoxCeleb & GRID & LRW    |         [PDF](https://arxiv.org/pdf/2107.09293.pdf)          |       [CODE](https://github.com/wangsuzhen/Audio2Head)       |
| Imitating Arbitrary Talking Style for Realistic Audio-Driven Talking Face Synthesis | ACMMM 2021 |        Ted-HD & LRW         |           [PDF](https://arxiv.org/abs/2111.00203)            |       [CODE](https://github.com/wuhaozhe/style_avatar)       |
| AD-NeRF: Audio Driven Neural Radiance Fields for Talking Head Synthesis | ICCV 2021  |              -              | [PDF](https://jixinya.github.io/projects/evp/resources/evp.pdf) |         [CODE](https://github.com/YudongGuo/AD-NeRF)         |
| FACIAL: Synthesizing Dynamic Talking Face with Implicit Attribute Learning | ICCV 2021  |              -              | [PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_FACIAL_Synthesizing_Dynamic_Talking_Face_With_Implicit_Attribute_Learning_ICCV_2021_paper.pdf) |       [CODE](https://github.com/zhangchenxu528/FACIAL)       |
| Learned Spatial Representations for Few-shot Talking-Head Synthesis | ICCV 2021  |          VoxCeleb           | [PDF](https://openaccess.thecvf.com/content/ICCV2021/papers/Meshry_Learned_Spatial_Representations_for_Few-Shot_Talking-Head_Synthesis_ICCV_2021_paper.pdf) |                              -                               |
| One-Shot Free-View Neural Talking-Head Synthesis for Video Conferencing | CVPR 2021  | VoxCeleb2 & TalkingHead-1KH | [PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_One-Shot_Free-View_Neural_Talking-Head_Synthesis_for_Video_Conferencing_CVPR_2021_paper.pdf) |                              -                               |
| Text2Video: Text-driven Talking-head Video Synthesis with Phonetic Dictionary | arXiv 2021 |          VidTIMIT           |           [PDF](https://arxiv.org/abs/2104.14631)            |       [CODE](https://github.com/sibozhang/Text2Video)        |

 

###  2020

| Title                                                        |   Venue    |            Datasets            |                             PDF                              |                            CODE                            |
| :----------------------------------------------------------- | :--------: | :----------------------------: | :----------------------------------------------------------: | :--------------------------------------------------------: |
| Realistic Face Reenactment via Self-Supervised Disentangling of Identity and Pose | AAAI 2020  |            VoxCeleb            |           [PDF](https://arxiv.org/abs/2003.12957)            |                             -                              |
| Robust One Shot Audio to Video Generation                    | CVPR 2020  |      GRID & LOMBARD GRID       | [PDF](https://openaccess.thecvf.com/content_CVPRW_2020/html/w45/Kumar_Robust_One_Shot_Audio_to_Video_Generation_CVPRW_2020_paper.html) |                             -                              |
| Learning Individual Speaking Styles for Accurate Lip to Speech Synthesis | CVPR 2020  |        GRID & TCD-TIMIT        | [PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Prajwal_Learning_Individual_Speaking_Styles_for_Accurate_Lip_to_Speech_Synthesis_CVPR_2020_paper.pdf) |                             -                              |
| Neural Voice Puppetry:  Audio-driven Facial Reenactment      | ECCV 2020  |               -                |         [PDF](https://arxiv.org/pdf/1912.05566.pdf)          | [CODE](https://github.com/JustusThies/NeuralVoicePuppetry) |
| Talking-head Generation with Rhythmic Head Motion            | ECCV 2020  | Crema & Grid & Voxceleb & Lrs3 |           [PDF](https://arxiv.org/abs/2007.08547)            |                             -                              |
| A Neural Lip-Sync Framework for Synthesizing Photorealistic Virtual News Anchors | ICPR 2020  |               -                |           [PDF](https://arxiv.org/abs/2002.08700)            |                             -                              |
| Talking Face Generation with Expression-Tailored Generative Adversarial Network | ACMMM 2020 |               -                |  [PDF](https://dl.acm.org/doi/abs/10.1145/3394171.3413844)   |                             -                              |
| A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild | ACMMM 2020 |              LRS2              |            [PDF](http://arxiv.org/abs/2008.10010)            |        [CODE](https://github.com/Rudrabha/Wav2Lip)         |

### Before 2020

| Title                                                        |   Venue    |            Datasets             |                             PDF                              |                             CODE                             |
| :----------------------------------------------------------- | :--------: | :-----------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| Few-Shot Adversarial Learning of Realistic Neural Talking Head Model | ICCV 2019  |            VoxCeleb             |           [PDF](https://arxiv.org/abs/1905.08233)            | [CODE](https://github.com/vincent-thevenin/Realistic-Neural-Talking-Head-Models) |
| Hierarchical Cross-Modal Talking Face Generation with Dynamic Pixel-Wise Loss | CVPR 2019  |           LRW & GRID            |  [PDF](http://www.cs.rochester.edu/u/lchen63/cvpr2019.pdf)   |        [CODE](https://github.com/lelechen63/ATVGnet)         |
| Talking Face Generation by Adversarially Disentangled Audio-Visual Representation | AAAI 2019  |               LRW               |           [PDF](https://arxiv.org/abs/1807.07860)            | [CODE](https://github.com/Hangz-nju-cuhk/Talking-Face-Generation-DAVS) |
| Realistic Speech-Driven Facial Animation with GANs           | IJCV 2019  | GRID & TCD-TIMIT & CREMA-D &LRW |            [PDF](http://arxiv.org/abs/1906.06337)            |                              -                               |
| Talking Face Generation by Conditional Recurrent Adversarial Network | IJCAI 2019 |   TCD-TIMIT & LRW & VoxCeleb    |           [PDF](https://arxiv.org/abs/1804.04786)            |  [CODE](https://github.com/susanqq/Talking_Face_Generation)  |
| Lip Movements Generation at a Glance                         | ECCV 2018  |         GRID &LRW &LDC          | [PDF](https://www.google.com.hk/url?sa=t&rct=j&q=&esrc=s&source=web&cd=3&cad=rja&uact=8&ved=2ahUKEwj54cbvupzoAhUyGKYKHXnfBuAQFjACegQIBBAB&url=http%3A%2F%2Fopenaccess.thecvf.com%2Fcontent_ECCV_2018%2Fpapers%2FLele_Chen_Lip_Movements_Generation_ECCV_2018_paper.pdf&usg=AOvVaw3FPJeIMPR56Bwm3k0bnQkI) |                              -                               |
| You said that?                                               | BMVC 2017  |         VoxCeleb & LRW          |           [PDF](https://arxiv.org/abs/1705.02966)            |                              -                               |



##  Datasets

• [LRS2](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html)

• [LRW](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html)

• [GRID](https://spandh.dcs.shef.ac.uk//avlombard/)

• [MEAD](https://wywu.github.io/projects/MEAD/MEAD.html)

• [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)

• [HDTF](https://github.com/MRzzm/HDTF)

• [SAVEE](http://kahlan.eps.surrey.ac.uk/savee/Download.html)

• [VOCA](https://voca.is.tue.mpg.de/)

• [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D)



##  Metrics

- PSNR (Peak Signal-to-Noise Ratio) : Measures the signal-to-noise ratio between the generated image and the original image, often used for comparing the similarity between two images. Higher PSNR values indicate better image quality.
- SSIM (Structural Similarity Index) : Evaluates the structural similarity between the generated image and the original image, considering brightness, contrast, and structure. SSIM values range from [-1, 1], with values closer to 1 indicating better image quality.
- LMD (Log-Mel Filterbank Distance) : Measures the Mel filterbank distance between the generated speech and the target speech. Lower LMD values indicate better speech generation quality.
- LRA (lip-reading accuracy) : used in evaluating speech generation quality, but focusing on the ratio of Mel filterbanks.
- FID (Fréchet inception distance) : Measures the quality of generated images by comparing the feature statistics of generated images to real images. Lower FID values indicate higher similarity between the distributions of generated and real images.
- LSE-D (Lip Sync Error - Distance) : Measures the error between the spectrogram of the generated speech and the real speech.
- LSE-C (Lip Sync Error - Confidence) :Similar to LSE-D, but considers a classifier model for measuring the error between the spectrogram of generated speech and real speech.
- LPIPS (Learned Perceptual Image Patch Similarity) : Utilizes a deep learning model to learn perceptual image quality, considering human perception of local image structures. Lower LPIPS values indicate better image generation quality.
- NIQE (Natural Image Quality Evaluator) : Used to evaluate the naturalness and quality of images, considering natural statistical properties. Lower NIQE values indicate better image quality.


##  Acknowledgements

This page was created by [Dan Zhao](https://github.com/SoberDanz), a graduate student at Dalian University of Technology.
