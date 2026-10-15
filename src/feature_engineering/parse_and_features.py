"""
parse_and_features.py

Input:
- data/raw/scam-dialogue_all.csv

Outputs:
- data/processed/scam_dialogue_parsed.csv
- data/processed/scam_dialogue_thread_features.csv

Run: 
- python src/feature_engineering/parse_and_features.py

"""

import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from config import RAW_DATA, PARSED_DATA, PROCESSED_DATA


MONEY_KEYWORDS = [
    "refund", "transfer", "pay", "voucher", "gift card",
    "fee", "amount", "bank", "account", "payment"
]

REQUEST_KEYWORDS = [
    "provide", "confirm", "send", "transfer", "give",
    "enter", "click", "call me", "verify",
    "confirm your", "please provide"
]

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}\b")
SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
AMOUNT_PATTERN = re.compile(
    r"\$\s*\d+(?:,\d{3})*(?:\.\d+)?|\b\d+(?:,\d{3})*(?:\.\d+)?\s*(USD|NZD|dollars|dollar|bucks)\b",
    flags=re.IGNORECASE,
)


def redact_pii(text):
    """
    Description:
        Redacts personally identifiable information from a message, including
        emails, phone numbers, SSN-like values, and money amounts.

    Input:
        text: A single message string from the dialogue.

    Output:
        A redacted version of the input text as a string.
    """
    text = str(text)
    text = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)
    text = PHONE_PATTERN.sub("[REDACTED_PHONE]", text)
    text = SSN_PATTERN.sub("[REDACTED_SSN]", text)
    text = AMOUNT_PATTERN.sub("[REDACTED_AMOUNT]", text)
    return text


def split_dialogue_into_turns(dialogue):
    """
    Description:
        Splits a full dialogue into individual ordered turns based on speaker
        prefixes such as caller: and receiver:.

    Input:
        dialogue: A full dialogue string from one dataset row.

    Output:
        A list of tuples in the format:
        [(speaker_role, message_text), ...]
    """
    dialogue = str(dialogue).strip()
    parts = re.split(r"(?=(?:caller:|receiver:))", dialogue, flags=re.IGNORECASE)

    turns = []

    for part in parts:
        part = part.strip()

        if not part:
            continue

        match = re.match(
            r"^(caller:|receiver:)\s*(.*)$",
            part,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if match:
            speaker = match.group(1).rstrip(":").lower()
            text = match.group(2).strip()
        else:
            speaker = "unknown"
            text = part

        turns.append((speaker, text))

    return turns


def is_request_message(text):
    """
    Description:
        Checks whether a message appears to contain a request using simple
        rule-based indicators such as question marks and request keywords.

    Input:
        text: A single message string.

    Output:
        True if the message is likely a request.
        False otherwise.
    """
    text = str(text).lower().strip()

    if text.endswith("?"):
        return True

    if any(keyword in text for keyword in REQUEST_KEYWORDS):
        return True

    if re.search(r"\b(can you|could you|please)\b", text):
        return True

    return False


def count_money_terms(text):
    """
    Description:
        Counts money-related signals in a message using predefined money
        keywords and amount patterns.

    Input:
        text: A single message string.

    Output:
        An integer count of money-related terms and amount patterns.
    """
    text = str(text)
    lower_text = text.lower()

    keyword_count = sum(lower_text.count(keyword) for keyword in MONEY_KEYWORDS)
    amount_count = len(AMOUNT_PATTERN.findall(text))

    return keyword_count + amount_count


def main():
    """
    Description:
        Runs the full parsing and feature extraction process. It reads the raw
        dataset, parses each dialogue into turns, redacts sensitive information,
        extracts structural thread-level features, and saves the processed CSV
        files.

    Input:
        No direct function input.
        Reads the raw dataset path from RAW_DATA in config.py.

    Output:
        Creates two CSV files:
        1. Parsed turn-level dataset saved to PARSED_DATA.
        2. Thread-level feature dataset saved to PROCESSED_DATA.
    """

    df = pd.read_csv(RAW_DATA)

    required_columns = {"dialogue", "label"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    parsed_rows = []
    feature_rows = []

    for index, row in df.iterrows():
        thread_id = f"thread_{index:06d}"
        raw_dialogue = row["dialogue"]
        label = row["label"]
        message_type = row.get("type", None)

        turns = split_dialogue_into_turns(raw_dialogue)

        if not turns:
            turns = [("unknown", raw_dialogue)]

        redacted_messages = []

        for turn_index, (speaker, text) in enumerate(turns):
            redacted_text = redact_pii(text)
            redacted_messages.append(redacted_text)

            parsed_rows.append(
                {
                    "thread_id": thread_id,
                    "message_id": f"{thread_id}_msg_{turn_index:03d}",
                    "turn_index": turn_index,
                    "speaker_role": speaker,
                    "message_text": redacted_text,
                    "thread_label": label,
                    "type": message_type,
                }
            )

        turn_count = len(redacted_messages)
        aggregate_num_urls = sum(len(URL_PATTERN.findall(message)) for message in redacted_messages)
        aggregate_money_terms = sum(count_money_terms(message) for message in redacted_messages)
        request_count = sum(1 for message in redacted_messages if is_request_message(message))
        ratio_request_messages = request_count / max(1, turn_count)

        feature_rows.append(
            {
                "thread_id": thread_id,
                "orig_index": index,
                "thread_label": label,
                "type": message_type,
                "turn_count": turn_count,
                "aggregate_num_urls": aggregate_num_urls,
                "aggregate_money_terms": aggregate_money_terms,
                "ratio_request_messages": ratio_request_messages,
            }
        )

    PARSED_DATA.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)

    parsed_df = pd.DataFrame(parsed_rows)
    features_df = pd.DataFrame(feature_rows)

    parsed_df.to_csv(PARSED_DATA, index=False)
    features_df.to_csv(PROCESSED_DATA, index=False)

    print(f"Saved parsed data: {PARSED_DATA}")
    print(f"Parsed rows: {len(parsed_df)}")

    print(f"Saved feature data: {PROCESSED_DATA}")
    print(f"Feature rows: {len(features_df)}")


if __name__ == "__main__":
    main()