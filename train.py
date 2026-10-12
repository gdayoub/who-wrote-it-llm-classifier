"""my cnn and lstm experiments for rq1 and rq2"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# first i check the csv and keep some questions away from training
# then i turn words into numbers because pytorch cant read raw text
# afteri train a cnn and an lstm to guess the writer
# finally i use validation to choose an epoch and test at the end


# this regex gives me words and punctuation as separate pieces
# i put r before the quotes so my backslashes work as regex characters
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
# i use this tuple to repeat rq2 with question text or answer text or both
MODES = ("input", "output", "both")
# this tuple lets me loop through both network types later
ARCHITECTURES = ("cnn", "lstm")


def split_text_into_tokens(text: str) -> list[str]:
    """my text becomes small pieces before i give them numbers"""
    # lower makes capitalized words match lowercase words in my vocabulary
    return TOKEN_RE.findall(text.lower())


def read_dataset(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """my csv gets checked before i spend time on training"""
    # with opens my csv and closes it for me when this block ends
    # dictreader lets me use csv headers as dictionary keys
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"LLM_name", "LLM_Input", "LLM_output", "prompt_id", "split"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Dataset needs LLM_name, LLM_Input, LLM_output, prompt_id, split")
        rows = list(reader)
    # this list comprehension keeps only my rows with required fields filled
    rows = [r for r in rows if all((r.get(k) or "").strip() for k in required)]
    # this set comprehension removes repeated model names for me
    # sorted keeps my label order the same on every run
    labels = sorted({r["LLM_name"] for r in rows})
    if len(labels) < 3:
        raise ValueError("The handout requires responses from at least three LLM families")
    if "family" in (reader.fieldnames or []):
        # i group family names by model id so i can catch mislabeled models
        families_by_label = defaultdict(set)
        for row in rows:
            families_by_label[row["LLM_name"]].add((row.get("family") or "").strip().casefold())
        if any(len(families) != 1 or "" in families for families in families_by_label.values()):
            raise ValueError("Every LLM_name needs one consistent, nonempty family label")
        if len({next(iter(families)) for families in families_by_label.values()}) < 3:
            raise ValueError("The handout requires at least three distinct LLM families")
    # i require both split names so i have a real held out test later
    if not {"train", "test"}.issubset({r["split"] for r in rows}):
        raise ValueError("Dataset needs both train and test rows")
    if any(r["split"] not in {"train", "test"} for r in rows):
        raise ValueError("Only train and test split values are supported")
    split_by_prompt: dict[str, str] = {}
    seen_pairs: set[tuple[str, str]] = set()
    for row in rows:
        # my dictionary remembers which split each prompt id belongs to
        # the same question in both groups would leak information into my test
        prompt_id, split = row["prompt_id"], row["split"]
        if prompt_id in split_by_prompt and split_by_prompt[prompt_id] != split:
            raise ValueError(f"Prompt {prompt_id} occurs in both train and test")
        split_by_prompt[prompt_id] = split
        pair = (prompt_id, row["LLM_name"])
        # my set remembers pairs because a set finds repeats quickly
        # repeats would give some of my answers extra weight during training
        if pair in seen_pairs:
            raise ValueError(f"Duplicate prompt/model pair: {pair}")
        seen_pairs.add(pair)
    for split in ("train", "test"):
        # i need all three writers in both groups for a fair score
        present = {r["LLM_name"] for r in rows if r["split"] == split}
        if present != set(labels):
            raise ValueError(f"Each LLM needs responses in {split}")
    labels_by_prompt = defaultdict(set)
    for row in rows:
        labels_by_prompt[row["prompt_id"]].add(row["LLM_name"])
    # i compare sets here to make sure each question has all three writers
    if any(found != set(labels) for found in labels_by_prompt.values()):
        raise ValueError("Each prompt must have one response from every selected LLM")
    return rows, labels


def train_val_test(rows: list[dict[str, str]], seed: int) -> tuple[list[int], list[int], list[int]]:
    """my questions stay together so my final test stays fair"""
    train_prompts = sorted({row["prompt_id"] for row in rows if row["split"] == "train"})
    if len(train_prompts) < 5:
        raise ValueError("At least five training prompts are needed for a separate validation set")
    # using the same seed gives me the same shuffled questions next time
    rng = random.Random(seed)
    rng.shuffle(train_prompts)
    # the slice takes about one fifth of my training prompts for validation
    # i choose an epoch with validation so the final test stays untouched
    val_prompts = set(train_prompts[:max(1, round(len(train_prompts) * 0.2))])
    # my lists hold row positions and not the actual row text
    training_row_ids = []
    validation_row_ids = []
    test_row_ids = []
    for row_index, row in enumerate(rows):
        # enumerate gives me both the row position and the row itself
        # i send all three answers to one prompt through the same branch
        if row["split"] == "test":
            test_row_ids.append(row_index)
        elif row["prompt_id"] in val_prompts:
            validation_row_ids.append(row_index)
        else:
            training_row_ids.append(row_index)
    return training_row_ids, validation_row_ids, test_row_ids


def text_for_experiment(row: dict[str, str], mode: str) -> str:
    """this chooses the text i let my model read"""
    # my if blocks return early so only one text choice comes back
    if mode == "input":
        return row["LLM_Input"]
    if mode == "output":
        return row["LLM_output"]
    # these words mark the boundary when i join question and answer text
    return "prompt: " + row["LLM_Input"] + " response: " + row["LLM_output"]


def build_training_vocabulary(rows: list[dict[str, str]], indices: list[int], mode: str, max_vocab: int) -> dict[str, int]:
    """my word to number dictionary comes from training text only"""
    # counter adds up how often i see each token in training rows
    # leaving test words out keeps my final test honest
    counts = Counter()
    for row_index in indices:
        training_text = text_for_experiment(rows[row_index], mode)
        counts.update(split_text_into_tokens(training_text))
    # i save zero for padding and one for words i did not learn
    # i start enumerate at two so real words get numbers two and above
    word_to_number = {}
    for number, (word, _) in enumerate(counts.most_common(max_vocab - 2), start=2):
        word_to_number[word] = number
    return word_to_number


def encode_rows_for_model(rows: list[dict[str, str]], indices: list[int], mode: str, vocab: dict[str, int],
           labels: list[str], max_length: int) -> TensorDataset:
    """my text and writer names become number tensors"""
    # my dictionary gives each writer a number from zero to two
    label_to_id = {label: number for number, label in enumerate(labels)}
    encoded_texts = []
    correct_labels = []
    for row_index in indices:
        row = rows[row_index]
        selected_text = text_for_experiment(row, mode)
        words = split_text_into_tokens(selected_text)
        # get gives me one when a word is missing from the dictionary
        # this slice keeps only my first max_length tokens
        word_numbers = [vocab.get(word, 1) for word in words]
        word_numbers = word_numbers[:max_length]
        # i add zeroes because pytorch batches need equal length lists
        # torch.tensor turns my python lists into data the network can use
        padding_needed = max_length - len(word_numbers)
        encoded_texts.append(word_numbers + [0] * padding_needed)
        correct_labels.append(label_to_id[row["LLM_name"]])
    return TensorDataset(torch.tensor(encoded_texts, dtype=torch.long), torch.tensor(correct_labels, dtype=torch.long))


class TextCNN(nn.Module):
    """my cnn uses short word patterns to guess the writer"""
    def __init__(self, vocab_size: int, classes: int, embedding_dim: int = 128):
        super().__init__()
        # embedding turns each word number into 128 values i can learn
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # modulelist registers my three conv layers with pytorch
        # the widths let me look at groups of 3 or 4 or 5 nearby tokens
        self.convs = nn.ModuleList(nn.Conv1d(embedding_dim, 64, width) for width in (3, 4, 5))
        # dropout hides some values during training so i memorize less
        self.dropout = nn.Dropout(0.3)
        # 64 signals from each width gives me 192 values for the final layer
        self.classifier = nn.Linear(64 * 3, classes)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """my batch of token numbers becomes writer scores"""
        # i transpose dimensions because conv1d wants channels before length
        embedded = self.embedding(token_ids).transpose(1, 2)
        # my for loop collects the strongest match from each conv layer
        # amax keeps my highest match across the whole text
        features = []
        for convolution in self.convs:
            activated = torch.relu(convolution(embedded))
            strongest_signal = activated.amax(dim=2)
            features.append(strongest_signal)
        return self.classifier(self.dropout(torch.cat(features, dim=1)))


class TextLSTM(nn.Module):
    """my lstm uses word order and memory to guess the writer"""
    def __init__(self, vocab_size: int, classes: int, embedding_dim: int = 128):
        super().__init__()
        # i use the same 128 value embedding size as my cnn for comparison
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # bidirectional reads forward and backward so i get both contexts
        # i keep 96 memory values from each direction
        self.lstm = nn.LSTM(embedding_dim, 96, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        # i join 96 plus 96 values before predicting one of three writers
        self.classifier = nn.Linear(192, classes)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """my real words get read in order before i predict a writer"""
        # ne means not equal so i count every token that is not zero padding
        lengths = token_ids.ne(0).sum(dim=1).clamp(min=1).cpu()
        embedded = self.embedding(token_ids)
        # packing tells my lstm where each answer stops before the zeroes
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        # hidden holds my last memory from each reading direction
        # negative indexes let me get the last two directions from that tensor
        _, (hidden, _) = self.lstm(packed)
        forward_memory = hidden[-2]
        backward_memory = hidden[-1]
        features = torch.cat((forward_memory, backward_memory), dim=1)
        return self.classifier(self.dropout(features))


@torch.no_grad()
def predict_model_labels(model: nn.Module, dataset: TensorDataset, batch_size: int, device: torch.device) -> list[int]:
    """this gets guesses without teaching my model from these rows"""
    # this decorator stops pytorch from saving gradients while i only predict
    # eval turns dropout off so my guesses stay consistent
    model.eval()
    predictions: list[int] = []
    # argmax gives me the position of the highest writer score
    for token_ids, _ in DataLoader(dataset, batch_size=batch_size):
        predictions.extend(model(token_ids.to(device)).argmax(dim=1).cpu().tolist())
    return predictions


def train_and_select_best_model(model: nn.Module, training: TensorDataset, validation: TensorDataset,
        epochs: int, batch_size: int, learning_rate: float, device: torch.device) -> tuple[nn.Module, int]:
    """my model practices then i restore its best version"""
    # to moves my model to the gpu when cuda is available
    model.to(device)
    # adamw is the rule i use for changing weights after a mistake
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # cross entropy compares my three scores with the correct writer number
    loss_fn = nn.CrossEntropyLoss()
    # i start below zero so the first real validation score always wins
    # best_state will hold my best model weights
    best_f1 = -1.0
    best_state = None
    best_epoch = 0
    labels = list(range(model.classifier.out_features))
    correct_validation_labels = validation.tensors[1].tolist()
    for epoch in range(1, epochs + 1):
        # range makes my epoch count from one through the requested total
        # one epoch means i show the model every training answer once
        model.train()
        total_training_loss = 0.0
        for token_ids, correct_labels in DataLoader(training, batch_size=batch_size, shuffle=True):
            # dataloader gives me small shuffled batches instead of all rows at once
            # zero_grad clears my last batch's weight change directions
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(token_ids.to(device)), correct_labels.to(device))
            # backward tells me how each weight affected this batch's loss
            loss.backward()
            # step changes my weights using those calculated gradients
            optimizer.step()
            total_training_loss += loss.item() * len(correct_labels)
        # i only make guesses on validation rows so their answers stay unseen
        validation_predictions = predict_model_labels(model, validation, batch_size, device)
        validation_f1 = f1_score(correct_validation_labels, validation_predictions, labels=labels, average="macro", zero_division=0)
        # dividing by len gives me an average loss per training answer
        average_training_loss = total_training_loss / len(training)
        # the f string puts live epoch numbers into my terminal output
        # a lower loss with flat validation f1 can mean i am memorizing
        print(f"  epoch {epoch}/{epochs} training_loss={average_training_loss:.3f} validation_f1={validation_f1:.3f}", flush=True)
        if validation_f1 > best_f1:
            # state_dict contains my learned weights
            # clone makes a real copy so later epochs cannot change my saved best
            best_f1 = validation_f1
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    assert best_state is not None
    # load_state_dict puts my best saved weights back into the model
    model.load_state_dict(best_state)
    return model, best_epoch


def run(args: argparse.Namespace) -> dict:
    """my six comparisons run here and i save what happened"""
    # these seeds give me repeatable splits and starting weights
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rows, labels = read_dataset(args.dataset)
    # train_val_test gives me row positions for each group of questions
    train_ids, val_ids, test_ids = train_val_test(rows, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using {device}; {len(train_ids)} train, {len(val_ids)} validation, {len(test_ids)} test rows")
    counts = defaultdict(Counter)
    # defaultdict makes a new counter when i reach a split for the first time
    # these counts show me if one writer has fewer answers
    for row in rows:
        counts[row["split"]][row["LLM_name"]] += 1
    # this dictionary records the settings i need to explain my experiment
    results: dict = {
        "dataset": str(args.dataset), "seed": args.seed, "labels": labels,
        "row_counts": {split: dict(values) for split, values in counts.items()},
        "split_prompt_ids": {
            "train": sorted({rows[i]["prompt_id"] for i in train_ids}),
            "validation": sorted({rows[i]["prompt_id"] for i in val_ids}),
            "test": sorted({rows[i]["prompt_id"] for i in test_ids}),
        },
        "device": str(device), "epochs_requested": args.epochs,
        "max_length": args.max_length, "max_vocab": args.max_vocab,
        "experiments": [],
    }
    predictions = []
    # this for loop lets me repeat training for question only and answer only and both
    for mode in MODES:
        # i rebuild the vocabulary because each mode contains different text
        vocab = build_training_vocabulary(rows, train_ids, mode, args.max_vocab)
        training = encode_rows_for_model(rows, train_ids, mode, vocab, labels, args.max_length)
        validation = encode_rows_for_model(rows, val_ids, mode, vocab, labels, args.max_length)
        testing = encode_rows_for_model(rows, test_ids, mode, vocab, labels, args.max_length)
        truth = testing.tensors[1].tolist()
        # this inner loop lets me run both networks on the same encoded rows
        for architecture in ARCHITECTURES:
            torch.manual_seed(args.seed)
            if architecture == "cnn":
                model = TextCNN(len(vocab) + 2, len(labels))
            else:
                model = TextLSTM(len(vocab) + 2, len(labels))
            print(f"training {architecture} with {mode} text", flush=True)
            model, best_epoch = train_and_select_best_model(model, training, validation, args.epochs,
                                    args.batch_size, args.learning_rate, device)
            # i use my held out test rows only after picking the best epoch
            predicted = predict_model_labels(model, testing, args.batch_size, device)
            # accuracy tells me what fraction of test guesses were right
            # macro f1 gives each of my writers equal weight in the score
            # the confusion matrix shows me which writers got mixed up
            result = {
                "architecture": architecture, "input_mode": mode,
                "best_epoch_by_validation_f1": best_epoch, "vocab_size": len(vocab) + 2,
                "test_accuracy": accuracy_score(truth, predicted),
                "test_macro_f1": f1_score(truth, predicted, labels=list(range(len(labels))), average="macro", zero_division=0),
                "confusion_matrix": confusion_matrix(truth, predicted, labels=list(range(len(labels)))).tolist(),
            }
            results["experiments"].append(result)
            print(f"{architecture:4} {mode:6} accuracy={result['test_accuracy']:.3f} macro_f1={result['test_macro_f1']:.3f}")
            # zip pairs my test row positions with their guessed writer numbers
            # saving every guess lets me inspect mistakes later
            for i, prediction in zip(test_ids, predicted):
                predictions.append({
                    "architecture": architecture, "input_mode": mode,
                    "prompt_id": rows[i]["prompt_id"], "true_label": rows[i]["LLM_name"],
                    "predicted_label": labels[prediction],
                })
    # i save the score summary and each test guess for my report
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (args.output / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["architecture", "input_mode", "prompt_id", "true_label", "predicted_label"])
        writer.writeheader()
        writer.writerows(predictions)
    return results


def main() -> None:
    """my terminal options get read here before i start the runs"""
    # argparse lets me change defaults by typing flags in the terminal
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/generated/responses.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/generated/experiments"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-vocab", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=448)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    # min checks the smallest setting so i can stop on bad values early
    if min(args.epochs, args.batch_size, args.max_vocab, args.max_length) < 1 or args.max_length < 5:
        parser.error("epochs, batch size, and vocabulary size must be positive; max length must be at least 5")
    try:
        run(args)
    except (OSError, ValueError) as exc:
        # try and except let me show a short error for a bad csv or path
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
