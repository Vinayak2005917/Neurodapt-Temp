import time
from functools import lru_cache
import torch

from transformers import AutoTokenizer, AutoModel

from Clause_Seg.model import ClauseSegmentationModel

from Clause_Seg.embeddings import _load_bert, text_to_embeddings, text_to_tokens

import re


@lru_cache(maxsize=None)
def _load_clause_model(model_path: str, device_name: str):
    model = ClauseSegmentationModel()
    state_dict = torch.load(model_path, map_location=device_name, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device_name)
    model.eval()
    return model

import re

import re


def post_processing(clauses, min_tokens=4):
    result = []

    for clause in clauses:
        clause = clause.strip()

        # Remove ##
        clause = re.sub(r"\s*##", "", clause)

        # Merge contractions
        clause = re.sub(r"\b(\w+)\s+'\s+(t|s|m|re|ve|ll|d)\b",r"\1'\2",clause)

        # Remove spaces before punctuation
        clause = re.sub(r"\s+([.,!?;:])", r"\1", clause)

        # Remove spaces around / and -
        clause = re.sub(r"\s*([/-])\s*", r"\1", clause)

        if not clause:
            continue

        # Merge punctuation-only clauses
        if re.fullmatch(r"[.,!?;:]+", clause):
            if result:
                result[-1] += clause
            else:
                result.append(clause)

        # Merge single-word clauses
        elif (result and re.fullmatch(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", clause)):
            result[-1] += " " + clause

        else:
            result.append(clause)
            
    changed = True

    while changed:
        changed = False

        for i, clause in enumerate(result):
            token_count = len(clause.split())

            if token_count >= min_tokens:
                continue

            # Only clause in the result
            if len(result) == 1:
                break

            # First clause -> merge right
            if i == 0:
                result[i + 1] = clause + " " + result[i + 1]
                result.pop(i)
                changed = True
                break

            # Last clause -> merge left
            if i == len(result) - 1:
                result[i - 1] += " " + clause
                result.pop(i)
                changed = True
                break

            left = result[i - 1]
            right = result[i + 1]

            # If left ends with a period, ALWAYS merge right
            if left.rstrip().endswith("."):
                result[i + 1] = clause + " " + right
                result.pop(i)
                changed = True
                break

            # Otherwise merge with the smaller neighbouring clause
            left_tokens = len(left.split())
            right_tokens = len(right.split())

            if left_tokens <= right_tokens:
                result[i - 1] = left + " " + clause
            else:
                result[i + 1] = clause + " " + right

            result.pop(i)
            changed = True
            break

    return result


class FullClauseSegmentation:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device)
        self.model = _load_clause_model(str(model_path), str(self.device))
        _load_bert(str(self.device))

    def predict_boundaries(self, embeddings):

        #now we send this to the model and get the logits
        with torch.no_grad():
            logits = self.model(embeddings)

        #apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        #get the predicted boundaries (1 if prob > 0.5 else 0)
        predicted_boundaries = (probs > 0.5).int().cpu().numpy()

        return predicted_boundaries

    def segment(self, text):
        # Empty dataset fields can occur in retellings.  BERT cannot run on
        # a zero-length token sequence, and there are no clauses to return.
        if not text or not text.strip():
            return []

        # Convert text to embeddings
        embeddings = text_to_embeddings(text, device=self.device)

        tokens = text_to_tokens(text, device=self.device)


        #now we send this to the model and get the logits
        logits = self.predict_boundaries(embeddings)

        #use the logits to print the claues, by matching the tokens and the predicted boundaries
        clauses = []
        current_clause = []
        for token, boundary in zip(tokens, logits[0]):
            if token.startswith("##"):
                boundary = 0
            current_clause.append(token)
            if boundary == 1:
                clauses.append(" ".join(current_clause))
                current_clause = []

        #if there are any remaining tokens in the current clause, add them as a clause
        if current_clause:
            clauses.append(" ".join(current_clause))

        #post processing of the clauses
        clauses = post_processing(clauses) 

        return clauses

        



if __name__ == "__main__":
    model_path = r"Clause_Seg\clause_segmentation_model_best.pt"
    text = "Me and my girlfriend had gone to the Los Angeles Zoo. I can't exactly remember the day but I believe it was either April or May or maybe June. It was a hot day that day and it was spur of the moment trip. I honestly wasn't particularly excited to go but my girlfriend had been bugging me for months to go to the zoo so I finally said let's go. We got there around 10 or 11 in the morning and it was already around 90 degrees. When we walked in there was the insect/reptile section of the zoo so we saw many exotic looking spiders, snakes, scorpions etc. I liked it but my girlfriend didn't like insects. Then we walked in the African animals section. We saw elephants mostly and we could hear monkeys and apes in the distance somewhere. So we keep walking and eventually we find the apes. We saw gorillas hanging out in the shade and in a different section we saw other chimps and monkeys. We walked past that and we saw giraffes. Watching the giraffes was a cool site because they got really close to us. Like one of them was looking at me and came in my direction and I thought it would actually do something to me. Then after we saw the lions and they were mostly hanging out in the shade to stay out of the heat. We continued to walk around and we saw a section with Australian animals. So we managed to see koalas in the tress and tasmanian devils. I think we also saw kangaroos but I can't remember. We spent a lot of time at the zoo maybe around 5-6 hours. I think we left and we got back home around 4 or 5 pm."
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clause_segmenter = FullClauseSegmentation(model_path, device=DEVICE)
    clauses = clause_segmenter.segment(text)
    print('='*25,"Story", "="*25)
    print(text)
    print('='*25,"Clauses", "="*25)
    count = 1
    for i in clauses:
        print(f"[{count}] {i}")
        count += 1

