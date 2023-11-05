from transformers import pipeline
from zeroShotLabels import labels


class ZeroShot:

    def model(self):
        zero_shot_classifier = pipeline("zero-shot-classification", device="cuda")
        return zero_shot_classifier

    def get_prediction(self, model, sentence, classes):
        # classi, classes, '', feture
        try:
            result = model(sequences=sentence, candidate_labels=classes, multi_label=False)
            predicted_classes = result['labels'][:1]
            predicted_score = result['scores'][:1]
            return predicted_classes[0], predicted_score[0]
        except Exception as e:
            print("Exception in getting prediction:", str(e))
            return "neutral", 0

    def moderationProcess(self, sentence, threshold):
        zero_shot_classifier = self.model()
        try:
            aa = self.get_prediction(zero_shot_classifier, sentence, labels.classes)
            confidence = aa[1] * 100
            print(aa[0], confidence)
            if confidence >= threshold:
                prediction = aa[0]
            else:
                prediction = "neutral"
        except Exception as e:
            print(e)
            prediction = "neutral"

        return prediction


