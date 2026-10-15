# Feature Dictionary

## Project

**Project Title:** Scam Conversation Risk Evaluator
**Dataset:** BothBosu scam-dialogue dataset
**Feature Level:** Thread-level, one row per conversation
**Purpose:** To support interpretable scam conversation detection using compact structural features.

This feature dictionary documents the engineered features extracted from each scam dialogue thread. The project intentionally uses compact structural indicators rather than TF-IDF, deep learning embeddings, or LLM-generated features. This supports transparency, reproducibility, and interpretability.

---

## Target Variable

### `thread_label`

**Description:**
The classification label for each conversation thread.

**Data Type:**
Integer / categorical

**Possible Values:**

* `0` = Non-scam conversation
* `1` = Scam conversation

**Source:**
Original `label` column from the raw dataset.

**Use in Modelling:**
This is the dependent variable used for supervised classification.

---

## Identifier Columns

### `thread_id`

**Description:**
A generated unique identifier for each conversation thread.

**Data Type:**
String

**Example:**
`thread_000001`

**Source:**
Generated during feature extraction from the row index of the raw dataset.

**Use in Modelling:**
Not used as a model feature. Used only for traceability and joining outputs.

---

### `orig_index`

**Description:**
The original row index of the conversation in the raw dataset.

**Data Type:**
Integer

**Valid Range:**
`0` to number of raw rows minus 1.

**Source:**
Original row position in the raw CSV file.

**Use in Modelling:**
Not used as a model feature. Used for reproducibility and saved train/test split tracking.

---

### `type`

**Description:**
The conversation type or category from the original dataset, if available.

**Data Type:**
String / categorical

**Source:**
Original `type` column from the raw dataset.

**Use in Modelling:**
Not included as one of the main structural model features. Retained for analysis and traceability.

---

## Engineered Structural Features

### `turn_count`

**Description:**
The total number of parsed dialogue turns in a conversation thread.

**Data Type:**
Integer

**Valid Range:**
Greater than `0`

**Calculation:**
The raw dialogue is split into individual speaker turns using speaker markers such as `caller:` and `receiver:`. The number of resulting turns is counted.

**Formula:**
`turn_count = number of parsed turns in the thread`

**Reason for Inclusion:**
Scam conversations may show different interaction lengths compared with non-scam conversations. Longer or more structured exchanges may indicate persuasion, repeated requests, or staged social engineering behaviour.

**Limitations:**
This feature depends on the quality of dialogue parsing. If speaker markers are inconsistent or missing, turn counts may be less accurate.

---

### `aggregate_num_urls`

**Description:**
The total number of URL patterns detected across all turns in a conversation thread.

**Data Type:**
Integer

**Valid Range:**
Greater than or equal to `0`

**Calculation:**
Each message is scanned using a URL regular expression. The total number of detected URLs is summed across the thread.

**Formula:**
`aggregate_num_urls = total URL matches across all messages in the thread`

**Reason for Inclusion:**
Scam conversations may include links that direct victims to fake websites, payment pages, credential harvesting forms, or malicious resources.

**Limitations:**
URLs are rare in the current dataset, so this feature may have limited predictive value. Its usefulness should be confirmed through model evaluation and ablation testing.

---

### `aggregate_money_terms`

**Description:**
The total number of money-related terms and amount patterns detected across all turns in a conversation thread.

**Data Type:**
Integer

**Valid Range:**
Greater than or equal to `0`

**Calculation:**
Each message is scanned for predefined money-related keywords and amount patterns. Counts are summed across all messages in the thread.

**Example Keywords:**
`refund`, `transfer`, `pay`, `voucher`, `gift card`, `fee`, `amount`, `bank`, `account`, `payment`

**Formula:**
`aggregate_money_terms = money keyword count + money amount pattern count`

**Reason for Inclusion:**
Many scams involve financial pressure, payments, refunds, bank accounts, gift cards, or transfer requests. A higher count of money-related terms may indicate increased scam risk.

**Limitations:**
Some legitimate conversations may also contain money-related words. This feature should not be interpreted alone without model context.

---

### `ratio_request_messages`

**Description:**
The proportion of messages in a thread that appear to contain a request.

**Data Type:**
Float

**Valid Range:**
`0.0` to `1.0`

**Calculation:**
Each message is checked using rule-based request indicators, including question marks and request-related keywords. The number of request-like messages is divided by the total number of turns.

**Example Request Keywords:**
`provide`, `confirm`, `send`, `transfer`, `give`, `enter`, `click`, `verify`, `please provide`

**Formula:**
`ratio_request_messages = request_message_count / turn_count`

**Reason for Inclusion:**
Scam conversations often involve repeated requests for information, verification, money transfer, account details, or action from the target. A high request ratio may indicate manipulative or coercive conversation patterns.

**Limitations:**
This is a heuristic feature. Some legitimate conversations may also contain many requests, and some scam messages may use indirect language that is not captured by the keyword list.

---

## Excluded Features

### TF-IDF Features

TF-IDF features were intentionally excluded from the final feature set. Although TF-IDF can improve text classification performance, it reduces interpretability and shifts the project away from compact structural indicators.

### Deep Learning Embeddings

Deep learning embeddings were excluded because the project focuses on lightweight, transparent, and reproducible feature engineering rather than high-dimensional representation learning.

### LLM-Based Features

LLM-generated features and LLM fine-tuning were excluded because the project does not include large language model training, deployment, or proprietary model dependence.

---

## Final Model Feature Set

The main model feature set contains:

1. `turn_count`
2. `aggregate_num_urls`
3. `aggregate_money_terms`
4. `ratio_request_messages`

These features are designed to support Logistic Regression and Decision Tree models while preserving interpretability, reproducibility, and auditability.
