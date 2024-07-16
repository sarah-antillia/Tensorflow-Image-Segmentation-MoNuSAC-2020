<h2>Tensorflow-Image-Segmentation-MoNuSAC-2020 (2024/07/17)</h2>

This is the first experiment of Image Segmentation for MoNuSAC 2020
(A Multi-organ Nuclei Segmentation and Classification Challenge) based on
the <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, 
and
<a href="https://drive.google.com/file/d/1YmhDmq_JLZKAp62uBUwXhDBTx0cVncQO/view?usp=sharing">
PreAugmented-MoNuSAC-ImageMask-Dataset-V2.zip (White-Mask)</a>, which was derived by us from the original dataset 
<a href="https://monusac-2020.grand-challenge.org/Data/">Challenges MoNuSAC 2020 Data</a>.<br>
<br>

The dataset used here has been taken from the following web-site<br>
<b>Challenges MoNuSAC 2020 Data</b><br>
<pre>
https://monusac-2020.grand-challenge.org/Data/
</pre>
<br>
On detail of the ImageMask Dataset, please refer to <a href="https://github.com/sarah-antillia/ImageMask-Dataset-MoNuSAC-2020">ImageMask-Dataset-MoNuSAC-2020</a>
<br>

<hr>
<b>Actual Image Segmentation</b><br>
The inferred masks predicted by our segmentation model trained on the MoNuSAC ImageMaskDataset appear similar 
to the ground truth masks, but lack precision in some areas. To improve segmentation accuracy, we could consider 
using a different segmentation model better suited for this task, or explore online data augmentation strategies.
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/mini_test/images/TCGA-DW-7963-01Z-00-DX1_5.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/mini_test/masks/TCGA-DW-7963-01Z-00-DX1_5.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/mini_test_output/TCGA-DW-7963-01Z-00-DX1_5.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/mini_test/images/TCGA-G9-6356-01Z-00-DX1_3.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/mini_test/masks/TCGA-G9-6356-01Z-00-DX1_3.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/mini_test_output/TCGA-G9-6356-01Z-00-DX1_3.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/mini_test/images/TCGA-G9-6367-01Z-00-DX1_4.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/mini_test/masks/TCGA-G9-6367-01Z-00-DX1_4.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/mini_test_output/TCGA-G9-6367-01Z-00-DX1_4.jpg" width="320" height="auto"></td>
</tr>

</table>

<hr>
<br>
In this experiment, we have used the simple UNet Model 
<a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this Oral Cancer Segmentation.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other Tensorflow UNet Models:<br>

<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<li><a href="./src/TensorflowDeepLabV3Plus.py">TensorflowDeepLabV3Plus.py</a></li>

<br>

<h3>1. Dataset Citatioin</h3>
The original dataset use here has been taken
<a href="https://monusac-2020.grand-challenge.org/Data/"
<b>Challenges/MoNuSAC 2020/Data</b></a><br>
<br>
<b>Data</b><br>
H&E staining of human tissue sections is a routine and most common protocol used by pathologists 
to enhance the contrast of tissue sections for tumor assessment (grading, staging, etc.) at multiple microscopic resolutions. Hence, we will provide the annotated dataset of H&E stained digitized tissue images of several patients acquired at multiple hospitals using one of the most common 40x scanner magnification. The annotations will be done with the help of expert pathologists. 
<br>
<br>
<b>License</b><br>
The challenge data is released under the creative commons license (CC BY-NC-SA 4.0).
<br>
<h3>
<a id="2">
2 MoNuSAC ImageMask Dataset
</a>
</h3>
 If you would like to train this MoNuSAC Segmentation model by yourself,
 please download the dataset from the google drive 
<a href="https://drive.google.com/file/d/1YmhDmq_JLZKAp62uBUwXhDBTx0cVncQO/view?usp=sharing">
PreAugmented-MoNuSAC-ImageMask-Dataset-V2.zip (White-Mask)</a>
<br>
Please expand the downloaded ImageMaskDataset and place it under <b>./dataset</b> folder to be
<pre>
./dataset
└─MoNuSAC
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>

<b>MoNuSAC Dataset Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/PreAugmented-MoNuSAC-ImageMask-Dataset-V2_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid dataset is not necessarily large. 
Probably, an online dataset augmentation strategy may be effective to improve segmentation accuracy.
<br>

<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
4 Train TensorflowUNet Model
</h3>
 We have trained MoNuSAC TensorflowUNet Model by using the following
<a href="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/MoNuSAC and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
This simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>
<pre>
; train_eval_infer.config
; 2024/07/16 (C) antillia.com

[model]
model          = "TensorflowUNet"
generator      = False
image_width    = 512
image_height   = 512
image_channels = 3
num_classes    = 1
input_normalize = False
normalization   = False

base_filters   = 16
base_kernels   = (7,7)
num_layers     = 8
dropout_rate   = 0.03
learning_rate  = 0.00005
clipvalue      = 0.3
dilation       = (1,1)
loss           = "bce_dice_loss"
metrics        = ["dice_coef"]

;metrics        = ["binary_accuracy"]
show_summary   = False

[train]
epochs        = 100
batch_size    = 2
patience      = 10
;metrics       = ["iou_coef", "val_iou_coef"]
;metrics       = ["binary_accuracy", "val_binary_accuracy"]
metrics       = ["dice_coef", "val_dice_coef"]

model_dir     = "./models"
eval_dir      = "./eval"
image_datapath = "../../../dataset/MoNuSAC/train/images/"
mask_datapath  = "../../../dataset/MoNuSAC/train/masks/"
create_backup  = False
learning_rate_reducer = True
reducer_factor        = 0.3
reducer_patience      = 4

save_weights_only = True

epoch_change_infer     = True
epoch_change_infer_dir =  "./epoch_change_infer"
num_infer_images       = 1

[eval]
image_datapath = "../../../dataset/MoNuSAC/valid/images/"
mask_datapath  = "../../../dataset/MoNuSAC/valid/masks/"

[test] 
image_datapath = "../../../dataset/MoNuSAC/test/images/"
mask_datapath  = "../../../dataset/MoNuSAC/test/masks/"

[infer] 
images_dir    = "./mini_test/images"
output_dir    = "./mini_test_output"
;merged_dir   = "./mini_test_output_merged"
;binarize      = True
;sharpening   = True

[segmentation]
colorize      = False
black         = "black"
white         = "green"
blursize      = None

[mask]
blur      = False
blur_size = (3,3)
binarize  = False
threshold = 127
</pre>
<hr>
<b>Model parameters</b><br>
Defined small <b>base_filters</b> and large <b>base_kernels</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
base_filters   = 16 
base_kernels   = (7,7)
num_layers     = 8
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation. To enable the augmentation, set generator parameter to True.  
<pre>
[model]
model         = "TensorflowUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "bce_dice_loss" and "dice_coef".<br>
<pre>
[model]
loss           = "bce_dice_loss"
metrics        = ["dice_coef"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback. 
<pre> 
[train]
learning_rate_reducer = True
reducer_factor        = 0.3
reducer_patience      = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>Epoch change inference callback</b><br>
Enabled EpochChange infer callback.<br>
<pre>
[train]
epoch_change_infer     = True
epoch_change_infer_dir =  "./epoch_change_infer"
num_infer_images       = 1
</pre>

By using this EpochChangeInference callback, on every epoch_change, the inference procedure can be called
 for an image in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/asset/epoch_change_infer.png" width="1024" height="auto"><br>
<br>
<br>
In this case, the training process stopped at epoch 35 by EarlyStopping Callback as shown below.<br>
<b>Training console output</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/asset/train_console_output_at_epoch_35.png" width="720" height="auto"><br>
<br>
<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
5 Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for MoNuSAC.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

<b>Evaluation console output</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/asset/evaluate_console_output_at_epoch_35.png" width="720" height="auto">
<br><br>

<a href="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/evaluation.csv">evaluation.csv</a><br>

The loss (bce_dice_loss) score for this test dataset is not so low, and dice_coef not so high as shown below.<br>
<pre>
loss,0.242
dice_coef,0.7487
</pre>

<h3>
6 Inference
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for MoNuSAC.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorflowUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>

<b>Enlarged Images and Masks Comparison</b><br>
<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/mini_test/images/TCGA-2Z-A9JN-01Z-00-DX1_1.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/mini_test/masks/TCGA-2Z-A9JN-01Z-00-DX1_1.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/mini_test_output/TCGA-2Z-A9JN-01Z-00-DX1_1.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/mini_test/images/TCGA-49-6743-01Z-00-DX2_001.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/mini_test/masks/TCGA-49-6743-01Z-00-DX2_001.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/mini_test_output/TCGA-49-6743-01Z-00-DX2_001.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/mini_test/images/TCGA-49-6743-01Z-00-DX2_004.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/mini_test/masks/TCGA-49-6743-01Z-00-DX2_004.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/mini_test_output/TCGA-49-6743-01Z-00-DX2_004.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/mini_test/images/TCGA-55-7573-01Z-00-DX1_001.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/mini_test/masks/TCGA-55-7573-01Z-00-DX1_001.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/mini_test_output/TCGA-55-7573-01Z-00-DX1_001.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/mini_test/images/TCGA-A2-A04X-01Z-00-DX1_002.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/mini_test/masks/TCGA-A2-A04X-01Z-00-DX1_002.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/MoNuSAC/mini_test_output/TCGA-A2-A04X-01Z-00-DX1_002.jpg" width="320" height="auto"></td>
</tr>
</table>

<br>
<br>


<h3>
References
</h3>
<b>1. Challenges MoNuSAC 2020 Data</b><br>
<pre>
https://monusac-2020.grand-challenge.org/Home/</pre>
<br>
<b>2. MoNuSAC2020: A Multi-Organ Nuclei Segmentation and Classification Challenge </b><br>
Ruchika Verma, Neeraj Kumar, Abhijeet Patil, Nikhil Cherian Kurian, Swapnil Rane, Simon Graham,<br>
Quoc Dang Vu, Mieke Zwager, Shan E Ahmed Raza, Nasir Rajpoot, Xiyi Wu, Huai Chen, Yijie Huang,<br>
Lisheng Wang, Hyun Jung, G Thomas Brown, Yanling Liu, Shuolin Liu, Seyed Alireza Fatemi Jahromi,<br>
Ali Asghar Khani, Ehsan Montahaei, Mahdieh Soleymani Baghshah, Hamid Behroozi, Pavel Semkin, <br>
Alexandr Rassadin, Prasad Dutande, Romil Lodaya, Ujjwal Baid, Bhakti Baheti, Sanjay Talbar, <br>
Amirreza Mahbod, Rupert Ecker, Isabella Ellinger, Zhipeng Luo, Bin Dong, Zhengyu Xu, Yuehan Yao,<br>
Shuai Lv, Ming Feng, Kele Xu, Hasib Zunair, Abdessamad Ben Hamza, Steven Smiley, Tang-Kai Yin,<br>
Qi-Rui Fang, Shikhar Srivastava, Dwarikanath Mahapatra, Lubomira Trnavska, Hanyun Zhang, <br>
Priya Lakshmi Narayanan, Justin Law, Yinyin Yuan, Abhiroop Tejomay, Aditya Mitkari, Dinesh Koka, <br>
Vikas Ramachandra, Lata Kini, Amit Sethi<br>
PMID: 34086562 DOI: 10.1109/TMI.2021.3085712<br>
<pre>
https://pubmed.ncbi.nlm.nih.gov/34086562/
</pre>

<br>
<b>3. ImageMask-Dataset-MoNuSAC-2020</b><br>
Toshiyuki Arai antillia.com<br>
<pre>
https://github.com/sarah-antillia/ImageMask-Dataset-MoNuSAC-2020
</pre>



