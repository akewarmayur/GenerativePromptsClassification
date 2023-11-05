import pandas as pd
results = pd.read_csv("results.csv")
statusRes = pd.DataFrame(columns=["prompt", "gt", "pred",
                                  "gtClassification",
                                  "classification", "subclassification",
"gtSubClassification",
                                  "status"])

did = {"violence": ["weapons",
                           "blood and gore",
                           "victim state",
                           "animal violence",
                           "physical violence"],
              "substance use": ["prescription", "smoking, tobacco, marijuana", "hard drugs", "alcohol"],
              "sexuality": ["intense", "mild", "same sex", "intense, full"],
              "nudity": ["full", "partial", "intense, full"],
       "profanity": ["profanity"],
       "hate": ["hate"],
       "neutral": ["neutral"]
              }

for ind, row in results.iterrows():
    if row['GT'] == row["prediction"]:
        st = "TP"
    elif row['GT'] == "neutral" and row["prediction"] != "neutral":
        st = "FP"
    elif row['GT'] != "neutral" and row["prediction"] == "neutral":
        st = "FN"
    else:
        st = "TN"

    GTclassification = "neutral"
    GTsubclassification = "neutral"
    ll = False
    for k, v in did.items():
        for ii in v:
            if row['GT'] == ii:
                GTclassification = k
                GTsubclassification = ii
                ll = True
                break
        if ll:
            break
    le = len(statusRes)
    statusRes.loc[le] = [row["prompt"], row["GT"], row["prediction"], GTclassification,row["classification"], GTsubclassification, row["subclassification"], st]

statusRes.to_csv("status7.csv")
