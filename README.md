# Who Wrote It?

**CMPSC 448 Midterm Project**

**Team leader:** George Dayoub  
**Team member:** Atharv Madne

## Project overview

This project tests whether a CNN or LSTM can identify which LLM generated a response. We created our own dataset by sending 100 prompts to three LLMs through the NVIDIA API. This produced 300 labeled responses covering coding math creative writing anime Elden Ring Penn State football and other topics.

## Research questions

1. Can we identify which LLM generated a response?
2. Does the user's prompt help identify the LLM?

## Models in the dataset

- GPT OSS (`openai/gpt-oss-20b`)
- Nemotron (`nvidia/nemotron-3.5-lightning-30b-a3b`)
- GLM (`z-ai/glm-5.3-flash`)

## Main results

Using only the generated responses the CNN reached 66.7% accuracy and the LSTM reached 41.7% accuracy. Prompt only classification stayed at the 33.3% baseline. Adding the prompt did not improve CNN accuracy and only slightly improved the LSTM.

## Project contents

- `collect.py` collects and saves responses from the three LLMs
- `train.py` runs the CNN and LSTM experiments for RQ1 and RQ2
- `data/` contains the prompts and collected dataset
- `report_materials/` contains the final graphs metrics and predictions
- `REPORT.typ` contains the editable report
- `REPORT.pdf` contains the submitted report



## Contact

- George Dayoub: Ged5222@psu.edu
- Atharv Madne: Adm6112@psu.edu
