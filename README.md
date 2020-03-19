# drug-discovery-for-COVID-19

[Coronavirus Deep Learning Competition](https://www.sage-health.org/)

Note: Due to limited amount of time to achieve deadline for submission of competition as the result of late acknowledgement of the competition, I will probably not be partaking in the competition but I will still continue on the research of the discovery for the drug using deep learning to contribute to the community in hopes that this application will be able to inspire or advance future progress in drug discoveries and technologies.

## Introduction
Siraj Raval has been very kind to share most of his content through YouTube and one of the content that Siraj Raval created was Coronavirus Deep learning Competition. The link for this competition video by Siraj Raval can be found [here](https://www.youtube.com/watch?v=1LJgkovowgA). 

To summarize the content from the video, viruses enter the human cells through binding of the virus' active site to the receptor of the human cells. As the virus' active site has good affinity with the receptor of humans' cell, this grants permission for the virus to enter the humans' cell to replicate itself and create more virus to attack the human body. In order for us to stop the virus from replicating and entering the human cell, we have to find candidate molecules that have high affinity with the active site of the virus to be able to bind to the active site of the virus to act as an inhibitor. Once the molecules have binded to the active site of the virus, the virus will not be able to bind to the receptor of human cells as their active site is being blocked by the candidate molecules.

To figure out how a molecule can bind to the active site of the virus, we can access to open sources dataset that provide the molecules in SMILES format. The dataset that I will be using can be found here [LSTMChem](https://github.com/topazape/LSTM_Chem/blob/master/datasets/dataset_cleansed.smi). The crystal structure of the Coronavirus can be found [here](https://www.wwpdb.org/pdb?id=pdb_00006lu7).

Once we have obtain the candidates molecules and the crystal structure of Coronavirus, We can use an AutoDocking software to determine the affinity of the candidate molecules to bind to the crystal structure of Coronavirus to act as an inhibitor. 
From Wikipedia, "docking is a method which predicts the preferred orientation of one molecule to a second when bound to each other to form a stable complex. Knowledge of the preferred orientation in turn may be used to predict the strength of association or binding affinity between two molecules using, for example, scoring functions." We will be able to determine how well a candidate molecule bind to the active site of the Coronavirus through AutoDocking software.

Examples of AutoDocking tools are Vina and PyRx (GUI), the candidate molecules will be called ligand and the Coronavirus will be called Macromolecules. In order to load ligand into PyRx, we require that the format of the ligand to be in pdbqt. As the dataset provided to us is in SMILES, we can use [OpenBabel](http://openbabel.org/wiki/Main_Page) to convert SMILES to pdbqt format.

```obabel -ismi "SMI_INPUT_FILENAME" --gen3d -opdbqt -O "OUTPUT_FILENAME" ```

We can use Vina to dock the ligand to the macromolecule by using:

``` vina --ligand {LIGAND_FILE.pdbqt} --macrmolecule {VIRUS_FILE.pdbqt} --center_x --center_y --center_z --size_x --size_y  --size_z ```

The center and the size value determine the search space to search to dock the ligand to the macromolecule.
To determine the center and size value, we can use PyRx to load the ligand and the macromolecule and generate the search space, PyRx will save them into a conf.txt file. 

With the conf.txt file, we can then run:

``` vina --ligand {LIGAND_FILE.pdbqt} --config conf.txt ```

vina will then output the affinity between the ligand and the macromolecule. The lower the affinity, the better the binding is.

<hr />

Currently, in Rajavithi Hospital from Thailand, the team have used influenza(oseltamivir) and HIV(lopinavir/ritonavir) drugs to improve patients with severe conditions from Coronavirus. 

In a paper by Taiwan as can be viewed [here](https://www.preprints.org/manuscript/202002.0242/v1/download), they discovered that indinavir has a better affinity with the Coronavirus than oseltamivir and lopinavir/ritonavir. The table can be seen here:
![Taiwan Research](https://github.com/darylfung96/drug-discovery-for-COVID-19/raw/master/image/taiwan%20research.png)
Table obtained from: doi: 10.20944/preprints202002.0242.v1


The goal of this project is to determine existing drugs that have higher affinity to bind to the active site of Coronavirus or develop a deep learning agent that will be able to generate a diversity of drugs that have high affinity to the active site of Coronavirus. 

### Plan

