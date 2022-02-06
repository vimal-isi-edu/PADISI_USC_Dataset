# PADISI-Finger Dataset
This document analyzes the PADISI-Finger dataset introduced in <img align="right" src="https://www.isi.edu/images/isi-logo.jpg" width="300"> [Multi-Modal Fingerprint Presentation Attack Detection: Evaluation On A New Dataset](https://arxiv.org/abs/2006.07498) and provides  
instructions on how to [download](#downloading-the-dataset) and [use](#using-the-dataset-no-custom-package-installation-required) the dataset. 
The provided dataset is only the *PADISI-USC* portion of the dataset. We are not authorized to release the 
*PADISI-APL* portion of the dataset, as mentioned in the manuscript.

## Dataset organization

### <ins>Ground truth file</ins>
The ground truth file is located under [data/finger_partitions/padisi_USC_FINGER_ground_truth.csv](./data/finger_partitions/padisi_USC_FINGER_ground_truth.csv). The `.csv` file contains the following columns:
* `transaction_id`: Alphanumeric code defining a single data collection session for a participant.
* `trial_id`: Alphanumeric code defining each captured finger of the participant.
* `trial_name`: One of `['FINGER_RIGHT_THUMB', 'FINGER_RIGHT_INDEX', 'FINGER_RIGHT_MIDDLE', 'FINGER_RIGHT_RING', 
                         'FINGER_LEFT_THUMB', 'FINGER_LEFT_INDEX', 'FINGER_LEFT_MIDDLE', 'FINGER_LEFT_RING']` specifying which finger was captured.
* `modality`: Biometric modality (`FINGER`).
* `ground_truth`: Ground truth code defined as:
    * `m00000`: Bona-fide
    * `m4<PAI_CODE><PAI_SPECIES>`: Presentation attack with `PAI_CODE = [01, 02, ..., 29]`, according to <em>**Table 1**</em> below.
       `PAI_SPECIES = [01, ..., MAX_PAI_SPECIES_FOR_CODE]`, according to <em>**Table 1**</em> below (e.g., `[01, ..., 04]` for `PAI_CODE = 01`). Example: `m40502` represents a presentation attack of the 2nd species of code `05`, which is a `Gummy material finger`.
       Note that only the *USC* columns of <em>**Table 1**</em> are relevant for the provided *PADISI-USC* portion of the dataset.
                               
* `participant_id`: Alphanumeric code uniquely defining each participant.
* `gender`, `ethnicity`, `race`, `age`: Demographic information about the participant for each sample (see statistics in <em>**Table 2**</em>).

### <ins>Partition files</ins>
All partition files are also located under [data/finger_partitions](./data/finger_partitions) with descriptive names. They involve all `3FOLD` and `LOO` protocols presented in the paper.
Each `.csv` file contains the following columns:
* `transaction_id`, `trial_id`, `trial_name`: Same as in the ground truth, [above](#insground-truth-fileins). Can be used to uniquely associate each row of a partition `.csv` file to a row in the ground truth `.csv` file.
* `partition_name`: One of `['train', 'valid', 'test']` specifying if a sample belongs to the `training`, `validation` or `testing` set, respectively.

<code>
<figcaption><b><em>Table 1</em></b> PAI counts in the collected datasets. For each PAI code, we provide a general PAI description, the number of different species as well as the attributes used for grouping PAI codes in terms of material, species, transparency, and attack type. PAI categories whose appearance depends heavily on the participant and preparation method are marked with *. Sponsor approval is required to release additional information about each PAI code.</figcaption></code>
<img src="https://github.com/ISICV/PADISI_USC_Dataset/blob/main/images/PADISI_datasets_info.png" width="700"/>

##
<code>
<figcaption><b><em>Table 2</em></b> <em>PADISI-USC</em> dataset statistics and demographics. </figcaption></code>
<img src="https://github.com/ISICV/PADISI_USC_Dataset/blob/main/images/PADISI_statistics_demographics.png" width="700"/>

## Downloading the dataset 

1. Please download and sign the provided [Data Transfer and Use Agreement](./documents/PADISI_USC_Finger_Data_Sharing_Agreement.pdf). 
Both the recipient (must be a project's principal investigator) and an official authorized signatory of the recipient’s organization must sign the agreement. 
For example, at a university, an authorized signatory is typically an administrative official, rather than a student or faculty member.
2. Submit the request and upload your **signed** Data Transfer and Use Agreement at [PADISI-USC Finger Dataset Request](https://docs.google.com/forms/d/e/1FAIpQLSfPAX9JbmehkbD4ss3zVal5cgfH1osCNNTDegY8PZBrfdui9w/viewform?vc=0&c=0&w=1&flr=0).
3. You will receive the download username/password and instructions upon approval and you can download the dataset within 30 days from approval.

If you have any questions about the data request process you can send an email to:

<img src="https://github.com/ISICV/PADISI_USC_Dataset/blob/main/images/e-mail.png" width="100"/> 
   
using `[PADISI USC Finger]: Dataset request question` on the subject line.

 
## Using the dataset (No custom package installation required)
For ease of use, we are providing a preprocessed version of our *PADISI-USC* dataset 
(used in the experiments presented in [Multi-Modal Fingerprint Presentation Attack Detection: Evaluation On A New Dataset](https://arxiv.org/abs/2006.07498)). When you download the preprocessed data, you will receive a 
file ```padisi_USC_FINGER_preprocessed.bz2``` which can be loaded in Python using the joblib package 
(see [conda installation](https://anaconda.org/anaconda/joblib) or 
[pip installation](https://joblib.readthedocs.io/en/latest/installing.html)) as:

```
import joblib
data_dict = joblib.load("padisi_USC_FINGER_preprocessed.bz2")
``` 

The above command will load a python dictionary with the following entries:

```
"data"       : 4d numpy array of shape (6211, 24, 160, 80)  
               - 6211: total number of samples
               -   24: total number of image channels per sample - each channel is normalized in 0-1 based on camera's bit depth (see paper for details).
               -  160: image height
               -   80: image width
"identifiers": list of 6211 string identifiers for all samples (unique string identifier per sample)
"labels"     : list of 6211 boolean labels for all samples
"pai_codes"  : list of 6211 string ground truth codes for all samples
```

* `"data"`: Contains all 24 channels of *F*<sub>*M*</sub>, *F*<sub>*S*</sub>, *F*<sub>*L*</sub>, and *B*<sub>*N*</sub> data, as presented in the paper:
    * *F*<sub>*L*</sub>: Channels `0-9` (10 frames of LSCI data - frames 10-19 out of 100 total frames).
    * *F*<sub>*S*</sub>: Channels `10-13` (4 images of SWIR data with order `[1200nm, 1300nm, 1450nm, 1550nm]`).
    * *F*<sub>*M*</sub>: Channels `14-20` (3 images of Visible and 4 of NIR data with order `[465nm, 591nm, white, 720nm, 780nm, 870nm, 940nm]`).
    * *B*<sub>*N*</sub>: Channels `21-23` (3 frames of Back-Illumination data - frames 10-12 out of 20 total frames).
* `"identifiers"`: Each identifier is constructed as `ID_<transaction_id>_<trial_id>_<trial_name>` using the entries of the ground truth or partition files [above](#insground-truth-fileins). This identifier can be used to uniquely associate each sample in the `"data"` with their corresponding ground truth or partition.
* `"labels"`: `False`: Bona-fide, `True`: Presentation attack.
* `"pai_codes"`: Ground truth code, using the rules discussed [above](#insground-truth-fileins).

## Example data loading (Requires installing custom packages)

We are providing a simple [example](./scripts/finger_scripts/finger_data_example.py) for looping through the 
preprocessed data using custom packages. The example also shows the creation of a [PyTorch DataLoader](https://pytorch.org/docs/stable/data.html). 
This requires creating an environment using [conda](https://docs.conda.io/en/latest/) and installing the provided package. 
The example code also works without PyTorch (if not installed).

   * **Installation instructions**:
        1. Use [Anaconda3](https://www.anaconda.com/products/individual) or [Miniconda3](https://docs.conda.io/en/latest/miniconda.html)
        2. Create the `padisi` environment: 
   
            ```conda env create -f padisi-environment.yml```
            
            or
            
            ```conda env create -f padisi-environment-with-pytorch.yml``` (if you want to use PyTorch).
            
        3. Activate the environment and install the package:
   
            ```
            conda activate padisi
            python setup.py install
            ```
   * **Running the example** (see [script](./scripts/finger_scripts/finger_data_example.py) for flag explanation and additional available flags):
   
        ```
        conda activate padisi
        python finger_data_example.py -dbp ../../data/finger_partitions/padisi_USC_FINGER_dataset_partition_3folds_part0.csv -extractor_id FM_FS
        ```

## Licence
The dataset and code is made available for academic or non-commercial purposes only.

USC MAKES NO EXPRESS OR IMPLIED WARRANTIES, EITHER IN FACT OR BY OPERATION OF LAW, 
BY STATUTE OR OTHERWISE, AND USC SPECIFICALLY AND EXPRESSLY DISCLAIMS ANY EXPRESS OR 
IMPLIED WARRANTY OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, VALIDITY OF 
THE SOFTWARE OR ANY OTHER INTELLECTUAL PROPERTY RIGHTS OR NON-INFRINGEMENT OF THE 
INTELLECTUAL PROPERTY OR OTHER RIGHTS OF ANY THIRD PARTY. SOFTWARE IS MADE AVAILABLE 
AS-IS. LIMITATION OF LIABILITY. TO THE MAXIMUM EXTENT PERMITTED BY LAW, IN NO EVENT 
WILL USC BE LIABLE TO ANY USER OF THIS CODE FOR ANY INCIDENTAL, CONSEQUENTIAL, 
EXEMPLARY OR PUNITIVE DAMAGES OF ANY KIND, LOST GOODWILL, LOST PROFITS, LOST BUSINESS 
AND/OR ANY INDIRECT ECONOMIC DAMAGES WHATSOEVER, REGARDLESS OF WHETHER SUCH DAMAGES 
ARISE FROM CLAIMS BASED UPON CONTRACT, NEGLIGENCE, TORT (INCLUDING STRICT LIABILITY 
OR OTHER LEGAL THEORY), A BREACH OF ANY WARRANTY OR TERM OF THIS AGREEMENT, AND 
REGARDLESS OF WHETHER USC WAS ADVISED OR HAD REASON TO KNOW OF THE POSSIBILITY OF 
INCURRING SUCH DAMAGES IN ADVANCE.
  
  
## Bibliography
If you use this dataset, please **cite the following publications**:

[1] [Leonidas Spinoulas](https://scholar.google.com/citations?user=SAw0POgAAAAJ&hl=en), 
    [Hengameh Mirzaalian](https://scholar.google.com/citations?user=BzaQhsoAAAAJ&hl=en), 
    [Mohamed Hussein](https://scholar.google.com/citations?hl=en&user=jCUt0o0AAAAJ), 
    and [Wael AbdAlmageed](https://scholar.google.com/citations?hl=en&user=tRGH8FkAAAAJ), 
    “[Multi-Modal Fingerprint Presentation Attack Detection: Evaluation On A New Dataset](https://ieeexplore.ieee.org/document/9399674)”, 
    in <em>IEEE Transactions on Biometrics, Behavior, and Identity Science</em>, 
    vol. 3, no. 3, pp. 347-364, July 2021, 
    doi: [10.1109/TBIOM.2021.3072325](https://doi.org/10.1109/TBIOM.2021.3072325)

[2] [Leonidas Spinoulas](https://scholar.google.com/citations?user=SAw0POgAAAAJ&hl=en), 
    [Mohamed Hussein](https://scholar.google.com/citations?hl=en&user=jCUt0o0AAAAJ), 
    [David Geissbühler](https://scholar.google.ch/citations?user=jbmrfWQAAAAJ&hl=fr), 
    Joe Mathai, 
    Oswin G. Almeida, 
    Guillaume Clivaz, 
    [Sébastien Marcel](https://scholar.google.com/citations?user=K9ku4jYAAAAJ&hl=en), 
    and [Wael AbdAlmageed](https://scholar.google.com/citations?hl=en&user=tRGH8FkAAAAJ), 
    “[Multispectral Biometrics System Framework: Application to Presentation Attack Detection](https://ieeexplore.ieee.org/document/9409166)”, 
    in <em>IEEE Sensors Journal</em>, 
    vol. 21, no. 13, pp. 15022-15041, July 2021,
    doi: [10.1109/JSEN.2021.3074406](https://doi.org/10.1109/JSEN.2021.3074406)


```
Bibtex format

@article{Spinoulas2021a,
  author    = {Spinoulas, Leonidas and 
               Mirzaalian, Hengameh and 
               Hussein, Mohamed E. and 
               AbdAlmageed, Wael},
  title     = {{Multi-Modal Fingerprint Presentation Attack Detection: Evaluation On A New Dataset}},
  journal   = {IEEE Transactions on Biometrics, Behavior, and Identity Science},
  year      = {2021},
  volume    = {3},
  number    = {3},
  pages     = {347-364},
  doi       = {10.1109/TBIOM.2021.3072325}
}

@article{Spinoulas2020b,
  author    = {Spinoulas, Leonidas and 
               Hussein, Mohamed E. and 
               Geissb{\"u}hler, David and 
               Mathai, Joe and 
               Almeida, Oswin G. 
               and Clivaz, Guillaume 
               and Marcel, S{\'e}bastien 
               and AbdAlmageed, Wael},
  title     = {{Multispectral Biometrics System Framework: Application to Presentation Attack Detection}},
  journal   = {IEEE Sensors Journal}, 
  year      = {2021},
  volume    = {21},
  number    = {13},
  pages     = {15022-15041},
  doi       = {10.1109/JSEN.2021.3074406}
}
```

## Acknowledgment

This research is based upon work supported by the Office of the Director of National 
Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA), via IARPA R\&D 
Contract No. 2017-17020200005. The views and conclusions contained herein are those of 
the authors and should not be interpreted as necessarily representing the official 
policies or endorsements, either expressed or implied, of the ODNI, IARPA, or the U.S. 
Government. The U.S. Government is authorized to reproduce and distribute reprints for 
Governmental purposes notwithstanding any copyright annotation thereon. 
