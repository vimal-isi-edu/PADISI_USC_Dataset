# PADISI-Face USC Dataset
This document analyzes the PADISI-Face dataset introduced in <img align="right" src="https://www.isi.edu/images/isi-logo.jpg" width="300"> [Detection and Continual Learning of Novel Face Presentation Attacks](https://openaccess.thecvf.com/content/ICCV2021/html/Rostami_Detection_and_Continual_Learning_of_Novel_Face_Presentation_Attacks_ICCV_2021_paper.html) and provides  
instructions on how to [download](#downloading-the-dataset) and [use](#using-the-dataset-no-custom-package-installation-required) the dataset. Additionally, it provides sentence descriptions for all available samples as introduced 
in [Explaining  Face  Presentation  Attack  Detection  Using  Natural  Language](https://arxiv.org/abs/2111.04862).  

## Dataset organization

### <ins>Ground truth file</ins>
The ground truth file is located under [data/face_partitions/padisi_USC_FACE_ground_truth.csv](./data/face_partitions/padisi_USC_FACE_ground_truth.csv). The `.csv` file contains the following columns:
* `transaction_id`: Alphanumeric code defining a single data collection session for a participant.
* `trial_id`: Alphanumeric code defining each independent capture of the participant.
* `trial_name`: String describing capture conditions (e.g., if participant is wearing glasses or makes a specific facial expression).
* `modality`: Biometric modality (`FACE`).
* `ground_truth`: Ground truth code defined as:
    * `m00000`: Bona-fide
    * `m5<PAI_CODE><PAI_SPECIES>`: Presentation attack with `PAI_CODE = [01, 02, ..., 09]`, according to <em>**Table 1**</em> below.
       `PAI_SPECIES = [01, ..., MAX_PAI_SPECIES_FOR_CODE]`, according to <em>**Table 1**</em> below (e.g., `[01, ..., 12]` for `PAI_CODE = 01`). Example: `m50602` represents a presentation attack of the 2nd species of code `06`, which is `Makeup`.
                               
* `participant_id`: Alphanumeric code uniquely defining each participant.
* `gender`, `ethnicity`, `race`, `age`: Demographic information about the participant for each sample.

### <ins>Partition files</ins>
All partition files are also located under [data/face_partitions](./data/face_partitions) with descriptive names. They involve three different `3FOLD` protocols.
Each `.csv` file contains the following columns:
* `transaction_id`, `trial_id`, `trial_name`: Same as in the ground truth, [above](#insground-truth-fileins). Can be used to uniquely associate each row of a partition `.csv` file to a row in the ground truth `.csv` file.
* `partition_name`: One of `['train', 'valid', 'test']` specifying if a sample belongs to the `training`, `validation` or `testing` set, respectively.

<code>
<figcaption><b><em>Table 1</em></b> PAI counts in the collected dataset. For each PAI code, we provide a general PAI description and the number of different species. Sponsor approval is required to release additional information about each PAI code.</figcaption></code>
<img src="https://github.com/ISICV/PADISI_USC_Dataset/blob/main/images/PADISI_Face_dataset_info.png" width="700"/>

##
<code>
<figcaption><b><em>Table 2</em></b> <em>PADISI-Face</em> dataset statistics and demographics. </figcaption></code>
<img src="https://github.com/ISICV/PADISI_USC_Dataset/blob/main/images/PADISI_Face_statistics_demographics.png" width="700"/>

### <ins>Text descriptions per sample<ins>
All text descriptions used in the experiments presented in [Explaining  Face  Presentation  Attack  Detection  Using  Natural  Language](https://arxiv.org/abs/2111.04862) can be found under
[data/face_partitions/padisi_USC_FACE_descriptions.csv](./data/face_partitions/padisi_USC_FACE_descriptions.csv). The provided `.csv` file contains the following columns:
* `transaction_id`, `trial_id`, `trial_name`: Same as in the ground truth, [above](#insground-truth-fileins). Can be used to uniquely associate each row of the descriptions `.csv` file to a row in the ground truth `.csv` file.
* `des0`, `des1`, `des2`, `des3`, `des4`: 5 freeform text descriptions for each sample. 


## Downloading the dataset 

1. Please download and sign the provided [Data Transfer and Use Agreement](./documents/PADISI_USC_Face_Data_Sharing_Agreement.pdf). 
Both the recipient (must be a project's principal investigator) and an official authorized signatory of the recipient’s organization must sign the agreement. 
For example, at a university, an authorized signatory is typically an administrative official, rather than a student or faculty member.
2. Submit the request and upload your **signed** Data Transfer and Use Agreement at [PADISI-USC Face Dataset Request](https://docs.google.com/forms/d/e/1FAIpQLScZSems8SsIzcJS6zGqeNn4khiLRYNeIza_HNO2odKyeMmmNA/viewform?usp=sf_link).
3. You will receive the download username/password and instructions upon approval and you can download the dataset within 30 days from approval.

If you have any questions about the data request process you can send an email to:

<img src="https://github.com/ISICV/PADISI_USC_Dataset/blob/main/images/e-mail.png" width="100"/> 
   
using `[PADISI USC Face]: Dataset request question` on the subject line.

 
## Using the dataset (No custom package installation required)
For ease of use, we are providing a preprocessed version of our *PADISI-Face* dataset 
(used in the experiments presented in [Detection and Continual Learning of Novel Face Presentation Attacks]). When you download the preprocessed data, you will receive a 
file ```padisi_USC_FACE_preprocessed.bz2``` which can be loaded in Python using the joblib package 
(see [conda installation](https://anaconda.org/anaconda/joblib) or 
[pip installation](https://joblib.readthedocs.io/en/latest/installing.html)) as:

```
import joblib
data_dict = joblib.load("padisi_USC_FACE_preprocessed.bz2")
``` 

The above command will load a python dictionary with the following entries:

```
"data"       : 4d numpy array of shape (2029, 3, 160, 80)  
               - 2029: total number of samples
               -    3: total number of image channels per sample - each channel is normalized in 0-1 based on camera's bit depth
               -  320: image height
               -  256: image width
"identifiers": list of 2029 string identifiers for all samples (unique string identifier per sample)
"labels"     : list of 2029 boolean labels for all samples
"pai_codes"  : list of 2029 string ground truth codes for all samples
```

* `"data"`: Contains all 3 color channels in order Blue, Green, Red (BGR format).
* `"identifiers"`: Each identifier is constructed as `ID_<transaction_id>_<trial_id>_<trial_name>` using the entries of the ground truth or partition files [above](#insground-truth-fileins). This identifier can be used to uniquely associate each sample in the `"data"` with their corresponding ground truth or partition.
* `"labels"`: `False`: Bona-fide, `True`: Presentation attack.
* `"pai_codes"`: Ground truth code, using the rules discussed [above](#insground-truth-fileins).

## Example data loading (Requires installing custom packages)

We are providing a simple [example](./scripts/face_scripts/face_data_example.py) for looping through the 
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
   * **Running the example** (see [script](./scripts/face_scripts/face_data_example.py) for flag explanation and additional available flags):
   
        ```
        conda activate padisi
        python face_data_example.py -dbp ../../data/face_partitions/padisi_USC_FACE_dataset_partition_3folds_part0.csv -extractor_id COLOR
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

[1] [Mohammad Rostami](https://scholar.google.ca/citations?user=Uzx8nLoAAAAJ&hl=en),
    [Leonidas Spinoulas](https://scholar.google.com/citations?user=SAw0POgAAAAJ&hl=en), 
    [Mohamed Hussein](https://scholar.google.com/citations?hl=en&user=jCUt0o0AAAAJ),
    [Joe Mathai](https://github.com/joemathai), 
    and [Wael AbdAlmageed](https://scholar.google.com/citations?hl=en&user=tRGH8FkAAAAJ), 
    ["Detection and Continual Learning of Novel Face Presentation Attacks"](https://openaccess.thecvf.com/content/ICCV2021/html/Rostami_Detection_and_Continual_Learning_of_Novel_Face_Presentation_Attacks_ICCV_2021_paper.html), 
    in <em>Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)</em>, 2021, pp. 14851-14860
    
[2] [Leonidas Spinoulas](https://scholar.google.com/citations?user=SAw0POgAAAAJ&hl=en), 
    [Mohamed Hussein](https://scholar.google.com/citations?hl=en&user=jCUt0o0AAAAJ), 
    [David Geissbühler](https://scholar.google.ch/citations?user=jbmrfWQAAAAJ&hl=fr), 
    [Joe Mathai](https://github.com/joemathai), 
    Oswin G. Almeida, 
    Guillaume Clivaz, 
    [Sébastien Marcel](https://scholar.google.com/citations?user=K9ku4jYAAAAJ&hl=en), 
    and [Wael AbdAlmageed](https://scholar.google.com/citations?hl=en&user=tRGH8FkAAAAJ), 
    ["Multispectral Biometrics System Framework: Application to Presentation Attack Detection"](https://ieeexplore.ieee.org/document/9409166), 
    in <em>IEEE Sensors Journal</em>, 
    vol. 21, no. 13, pp. 15022-15041, July 2021,
    doi: [10.1109/JSEN.2021.3074406](https://doi.org/10.1109/JSEN.2021.3074406)

If you further use the [text descriptions](#instext-descriptions-per-sampleins), please **cite the following publication**:

[3] [Hengameh Mirzaalian](https://scholar.google.ca/citations?user=BzaQhsoAAAAJ&hl=en),
    [Mohamed Hussein](https://scholar.google.com/citations?hl=en&user=jCUt0o0AAAAJ),
    [Leonidas Spinoulas](https://scholar.google.com/citations?user=SAw0POgAAAAJ&hl=en),
    [Jonathan May](https://scholar.google.com/citations?user=tmK5EPEAAAAJ&hl=en),
    and [Wael AbdAlmageed](https://scholar.google.com/citations?hl=en&user=tRGH8FkAAAAJ),
    ["Explaining  Face  Presentation  Attack  Detection  Using  Natural  Language"](https://arxiv.org/abs/2111.04862), in <em>IEEE International Conference on Automatic Face and Gesture Recognition</em>, 2021


```
Bibtex format

@InProceedings{Rostami_2021_ICCV,
    author    = {Rostami, Mohammad and Spinoulas, Leonidas and Hussein, Mohamed and Mathai, Joe and Abd-Almageed, Wael},
    title     = {Detection and Continual Learning of Novel Face Presentation Attacks},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {14851-14860}
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

@misc{mirzaalian2021explaining,
      title={Explaining Face Presentation Attack Detection Using Natural Language}, 
      author={Hengameh Mirzaalian and 
              Mohamed E. Hussein and 
              Leonidas Spinoulas and 
              Jonathan May and 
              Wael Abd-Almageed},
      year={2021},
      eprint={2111.04862},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
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
