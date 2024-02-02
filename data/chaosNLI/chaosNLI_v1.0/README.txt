[C]ollective [H]um[A]n [O]pinion[S] on [N]atural [L]anguage [I]nference Data: ChaosNLI

Creator: Yixin Nie (nyixin318@gmail.com)

ChaosNLI is a dataset having 100 annotations on each examples for the development set of three oft-used NLI dataset: SNLI, MNLI, alphaNLI.
The main purpose of ChaosNLI is to study collective human opinions on Natural Language Inference data and how model can capture the human opinion distributions.

Statistics
ChaosNLI-SNLI:      1,514
ChaosNLI-MNLI:      1,599
ChaosNLI-alphaNLI:  1,532
A total of 464,500 annotations.

ChaosNLI is licensed under Creative Commons-Non Commercial 4.0.
Please find more details in: https://github.com/easonnie/chaos_nli

"entailment": "e"
"neutral": "n"
"contradiction": "c"

Sample (ChaosNLI-SNLI):
{   "uid": "193596775.jpg#3r1n",
    "label_counter": {"n": 67, "c": 29, "e": 4},
    "majority_label": "n",
    "label_dist": [0.04, 0.67, 0.29],
    "label_count": [4, 67, 29],
    "entropy": 1.0907619435810212,
    "example": {
        "uid": "193596775.jpg#3r1n",
        "premise": "A woman is talking on the phone while standing next to a dog.",
        "hypothesis": "A woman is walking her dog.",
        "source": "snli_agree_3"
    },
    "old_label": "n",
    "old_labels": ["neutral", "neutral", "neutral", "contradiction", "contradiction"]
}

Sample (ChaosNLI-MNLI):
{   "uid": "132539e",
    "label_counter": {"c": 14, "n": 36, "e": 50},
    "majority_label": "e",
    "label_dist": [0.5, 0.36, 0.14],
    "label_count": [50, 36, 14],
    "entropy": 1.4277254052800654,
    "example": {
        "uid": "132539e",
        "premise": "Boca da Corrida Encumeada (moderate; 5 hours): views of Curral das Freiras and the valley of Ribeiro do Poco.",
        "hypothesis": "Boca da Corrida Encumeada is a moderate text that takes 5 hours to complete. ",
        "source": "mnli_agree_3"},
    "old_label": "e",
    "old_labels": ["entailment", "entailment", "neutral", "entailment", "contradiction"]
}

Sample (ChaosNLI-alphaNLI):
{   "uid": "066eff23-d80a-4fc3-adf2-794948d7926a-1",
    "label_counter": {"2": 63, "1": 37},
    "majority_label": 2,
    "label_dist": [0.37, 0.63],
    "label_count": [37, 63],
    "entropy": 0.950672092687066,
    "example": {
        "uid": "066eff23-d80a-4fc3-adf2-794948d7926a-1",
        "obs1": "Stan started to feel sick at school one day.",
        "obs2": "Stan finally recovered but said he wanted a flu shot from now on.",
        "hyp1": "Stan was out of school for a week with the stomach ache.",
        "hyp2": "The school nurse sent Stan home from school.",
    "source": "abdnli_dev"},
    "old_label": 2
}