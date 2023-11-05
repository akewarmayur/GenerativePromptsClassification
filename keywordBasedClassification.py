import pandas as pd
import difflib
from keywords import moderationKeywords


class Keyword:

    def __int__(self):
        self.di = {"violence": ["weapons",
                           "blood and gore",
                           "victim state",
                           "animal violence",
                           "physical violence"],
              "substance use": ["prescription", "smoking, tobacco, marijuana", "hard drugs", "alcohol"],
              "sexuality": ["intense", "mild", "same sex", "intense, full"],
              "nudity": ["full", "partial", "intense, full"],
                   "profanity": ["profanity"],
                   "hate": ["hate"],
                   "neutral":["neutral"]
              }

    def matchKeyWords(self, par1, par2):
        # compare
        return difflib.SequenceMatcher(None, par1, par2).ratio() * 100

    def startProcess(self, sentence):
        words_list = sentence.split(" ")
        predictions = []
        for key, value in moderationKeywords.keywords_dict.items():
            for w in words_list:
                for v in value:
                    per = self.matchKeyWords(w.lower(), v.lower())
                    if per >= 80:
                        predictions.append(key)
                        break
        return predictions


res = pd.DataFrame(columns=["prompt", "GT", "prediction", "classification", "subclassification"])
df = pd.read_csv("data/PromptsDataTrain.csv")
print(df.columns)
obj = Keyword()
did = {"violence": ["weapons",
                           "blood and gore",
                           "victim state",
                           "animal violence",
                           "physical violence"],
              "substance use": ["prescription", "smoking, tobacco, marijuana", "hard drugs", "alcohol"],
              "sexuality": ["intense", "mild", "same sex", "intense, full"],
              "nudity": ["full", "partial", "intense, full"],
                "profanity":["profanity"],
       "hate":["hate"],
       "neutral": ["neutral"]
              }
for ind, row in df.iterrows():
    ss = obj.startProcess(row['prompt'])
    print(ind, ss)
    if len(ss) >= 1:
        pred = ss[0]
        classification = "neutral"
        subclassification = "neutral"
        ll = False
        for k, v in did.items():
            for ii in v:
                if pred == ii:
                    classification = k
                    subclassification = ii
                    ll = True
                    break
            if ll:
                break
    else:
        pred = "neutral"
        classification = "neutral"
        subclassification = "neutral"

    le = len(res)
    res.loc[le] = [row['prompt'], row["finalLabel"], pred, classification, subclassification]
res.to_csv("results.csv")