# Computational Model for Episodic Timeline Based on Multiscale Synaptic Decay

### Authors: James Mochizuki-Freeman, Sara Zomorodi, Sahaj Singh Maini, and Zoran Tiganj

## Abstract
> Human episodic memory enables the retrieval of temporally organized past experiences. Retrieval cues can target both semantic content and temporal location, reflecting the multifaceted nature of episodic recall. Existing computational models provide mechanistic accounts for how temporally organized memories of the recent past (seconds to minutes) can persist in neural activity. However, it remains unclear how episodic memories can be stored and retrieved while preserving temporal structure within and across episodes. Here, we propose a computational model that uses a spectrum of synaptic decay rates to store temporally organized memories of the recent past as an episodic timeline. We characterize how the memories can be retrieved using either a memory of the recent past, specific semantic cues, or temporal addressing. This approach thus bridges short-term working memory and longer-term episodic storage, offering a computational model of how synaptic dynamics can maintain temporally structured events.

## Setup instructions
Experiments were run with Python 3.13. Use the following commands to set up the environment:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the code
To run the code, install the previously created virtual environment and start Jupyter Notebook:

```bash
ipython kernel install --user --name=venv
jupyter notebook
```


