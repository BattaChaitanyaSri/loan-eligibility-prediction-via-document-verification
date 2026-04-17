import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Load model for semantic similarity
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

def load_data(file):
    y_true, y_pred = [], []
    ref_texts, pred_texts = [], []

    with open(file, "r") as f:
        for line in f:
            data = json.loads(line)

            y_true.append(data["decision"])
            y_pred.append(data.get("predicted_decision", data["decision"]))  # replace with model output

            ref_texts.append(data["reasoning"])
            pred_texts.append(data.get("predicted_reasoning", data["reasoning"]))

    return y_true, y_pred, ref_texts, pred_texts


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label="Approved")
    recall = recall_score(y_true, y_pred, pos_label="Approved")
    f1 = f1_score(y_true, y_pred, pos_label="Approved")

    return acc, precision, recall, f1


def compute_rouge(refs, preds):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = [scorer.score(r, p)['rougeL'].fmeasure for r, p in zip(refs, preds)]
    return np.mean(scores)


def compute_meteor(refs, preds):
    scores = [meteor_score([r.split()], p.split()) for r, p in zip(refs, preds)]
    return np.mean(scores)


def compute_semantic_similarity(refs, preds):
    embeddings1 = bert_model.encode(refs, convert_to_tensor=True)
    embeddings2 = bert_model.encode(preds, convert_to_tensor=True)

    scores = util.cos_sim(embeddings1, embeddings2).diagonal()
    return float(scores.mean())


def evaluate(file):
    y_true, y_pred, refs, preds = load_data(file)

    acc, precision, recall, f1 = compute_metrics(y_true, y_pred)
    rouge = compute_rouge(refs, preds)
    meteor = compute_meteor(refs, preds)
    semantic = compute_semantic_similarity(refs, preds)

    print("\n📊 Evaluation Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROUGE-L: {rouge:.4f}")
    print(f"METEOR: {meteor:.4f}")
    print(f"Semantic Similarity: {semantic:.4f}")


if __name__ == "__main__":
    evaluate("P:\loan prediction via sml\training\test_derived_samples.jsonl")