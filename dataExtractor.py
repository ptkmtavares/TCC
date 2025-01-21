import logging
from email.parser import HeaderParser
import os
import re
import email.utils as emailUtils
import random
import concurrent.futures
from typing import List, Tuple, Dict, Union
from receivedParser import ReceivedParser
from dataOrganizer import main as organize_data

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
parser = HeaderParser()
receivedParser = ReceivedParser()

FEATURES = [
    "time_zone",
    "date_date_received_diff",
    "missing_importance",
    "missing_x-mailman-version",
    "missing_user-agent",
    "missing_x-mailer",
    "missing_list-unsubscribe",
    "missing_mime-version",
    "missing_references",
    "missing_x-original-to",
    "missing_domainkey-signature",
    "missing_received-spf",
    "missing_dmarc",
    "str_content-encoding_empty",
    "str_from_question",
    "str_from_exclam",
    "str_from_chevron",
    "str_to_chevron",
    "str_to_undisclosed",
    "str_to_empty",
    "str_message-ID_dollar",
    "str_return-path_bounce",
    "str_return-path_empty",
    "str_reply-to_question",
    "str_content-type_texthtml",
    "str_precedence_list",
    "str_received-SPF_bad",
    "str_received-SPF_softfail",
    "str_received-SPF_fail",
    "str_dmarc_bad",
    "str_dmarc_softfail",
    "str_dmarc_fail",
    "str_dkim_bad",
    "str_dkim_softfail",
    "str_dkim_fail",
    "received_str_forged",
    "email_match_from_reply-to",
    "domain_val_message-id",
    "domain_match_message-id_from",
    "domain_match_message-id_return-path",
    "domain_match_message-id_sender",
    "domain_match_message-id_reply-to",
    "domain_match_from_return-path",
    "domain_match_message-id_from",
    "domain_match_from_return-path",
    "domain_match_message-id_return-path",
    "domain_match_message-id_sender",
    "domain_match_message-id_reply-to",
    "domain_match_return-path_reply-to",
    "domain_match_reply-to_to",
    "domain_match_to_in-reply-to",
    "domain_match_errors-to_message-id",
    "domain_match_errors-to_from",
    "domain_match_errors-to_sender",
    "domain_match_errors-to_reply-to",
    "domain_match_sender_from",
    "domain_match_references_reply-to",
    "domain_match_references_in-reply-to",
    "domain_match_references_to",
    "domain_match_from_reply-to",
    "domain_match_to_from",
    "domain_match_to_message-id",
    "domain_match_to_received",
    "domain_match_reply-to_received",
    "domain_match_from_received",
    "domain_match_return-path_received",
]

HEADER_INFORMATION = [
    "label",
    "received_hop_1",
    "received_hop_2",
    "received_hop_3",
    "received_hop_4",
    "received_hop_5",
    "received_hop_6",
    "received_hop_7",
    "received_hop_8",
    "received_hop_9",
    "received_hop_10",
    "received_hop_11",
    "received_hop_12",
    "received_hop_13",
    "received_hop_14",
    "received_hop_15",
    "received_hop_16",
    "from",
    "date",
    "hop_count",
    "subject",
    "message-id",
    "to",
    "content-type",
    "mime-version",
    "x-mailer",
    "content-transfer-encoding",
    "x-mimeole",
    "x-priority",
    "return-path",
    "list-id",
    "lines",
    "x-virus-scanned",
    "status",
    "content-length",
    "precedence",
    "delivered-to",
    "list-unsubscribe",
    "list-subscribe",
    "list-post",
    "list-help",
    "x-msmail-priority",
    "x-spam-status",
    "sender",
    "errors-to",
    "reply-to",
    "x-beenthere",
    "list-archive",
    "x-mailman-version",
    "x-miltered",
    "x-uuid",
    "x-virus-status",
    "x-spam-level",
    "x-spam-checker-version",
    "references",
    "user-agent",
    "received-spf",
    "in-reply-to",
    "x-original-to",
    "user-agent",
    "arc-message-signature",
    "arc-authentication-results",
    "arc-seal",
    "thread-index",
    "cc",
    "content-disposition",
    "mailing-list",
    "x-spam-check-by",
    "domainkey-signature",
    "dkim-signature",
    "importance",
    "x-mailing-list",
]


def __extract_emails(text: str) -> List[str]:
    """Extracts email addresses from a given text.

    Args:
        text (str): The text to extract email addresses from.

    Returns:
        List[str]: A list of extracted email addresses.
    """
    if not text:
        return []
    in_brackets = re.findall(
        r"<([a-zA-Z0-9+._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z0-9_-]+)>", text
    )
    if not in_brackets:
        not_in_brackets = re.findall(
            r"([a-zA-Z0-9+.-]+@[a-zA-Z0-9.-]+\.[a-zA-Z0-9_-]+)", text
        )
        return not_in_brackets if not_in_brackets else []
    return in_brackets


def __email_same_check(emails1: List[str], emails2: List[str]) -> int:
    """Checks if there are any matching email addresses between two lists.

    Args:
        emails1 (List[str]): The first list of email addresses.
        emails2 (List[str]): The second list of email addresses.

    Returns:
        int: 1 if there is a match, 0 if no match, -1 if either list is empty.
    """
    if not emails1 or not emails2:
        return -1
    for email1 in emails1:
        for email2 in emails2:
            if email1 == email2:
                return 1
    return 0


def __extract_domains(text: str) -> List[str]:
    """Extracts domains from email addresses found in a given text.

    Args:
        text (str): The text to extract domains from.

    Returns:
        List[str]: A list of extracted domains.
    """
    emails_list = __extract_emails(text)
    if not emails_list:
        return []
    domains_list = []
    for email in emails_list:
        if len(email.split(".")) < 2:
            continue
        main_domain = email.split("@")[-1]
        main_domain = main_domain.split(".")[-2:]
        main_domain = main_domain[0] + "." + re.sub("\W+", "", main_domain[1])
        domains_list.append(main_domain.lower())
    return domains_list


def __domain_match_check(domains1: List[str], domains2: List[str]) -> int:
    """Checks if there are any matching domains between two lists.

    Args:
        domains1 (List[str]): The first list of domains.
        domains2 (List[str]): The second list of domains.

    Returns:
        int: 1 if there is a match, 0 if no match, -1 if either list is empty.
    """
    if not domains1 or not domains2:
        return -1
    for d1 in domains1:
        for d2 in domains2:
            if d1 == d2:
                return 1
    return 0


def __check_shady_domain(domains: List[str]) -> int:
    """Checks if any domain in the list is considered shady.

    Args:
        domains (List[str]): The list of domains to check.

    Returns:
        int: 1 if a shady domain is found, 0 otherwise, -1 if the list is empty.
    """
    if not domains:
        return -1
    for domain in domains:
        if "uwaterloo.ca" in domain:
            return 1
    return 0


def __get_str_features(
    email_info: Dict[str, str],
    info_name: str,
    feature_list: Dict[str, int],
    features: List[str],
    conditions_check: List[str],
) -> None:
    """Extracts string-based features from email information.

    Args:
        email_info (Dict[str, str]): The email information dictionary.
        info_name (str): The name of the information field to check.
        feature_list (Dict[str, int]): The dictionary to store extracted features.
        features (List[str]): The list of feature names.
        conditions_check (List[str]): The list of conditions to check for each feature.
    """
    for feature_name, condition_check in zip(features, conditions_check):
        if condition_check == "":
            feature_list[feature_name] = (
                1 if re.match(condition_check, email_info[info_name]) else 0
            )
        else:
            feature_list[feature_name] = (
                1
                if re.search(condition_check, email_info[info_name], re.IGNORECASE)
                else 0
            )


def __get_time_zone(email_info: Dict[str, str]) -> int:
    """Determines the time zone from the email date.

    Args:
        email_info (Dict[str, str]): The email information dictionary.

    Returns:
        int: 0 if the time zone is UTC-4, 1 otherwise, -1 if parsing fails.
    """
    time_zone = emailUtils.parsedate_tz(email_info["date"])
    if time_zone is None:
        return -1
    return 0 if (int(time_zone[9] / (60 * 60)) % 24) == 20 else 1


def __get_date_date_received_diff(email_info: Dict[str, str]) -> int:
    """Calculates the difference between the email date and the received date.

    Args:
        email_info (Dict[str, str]): The email information dictionary.

    Returns:
        int: 0 if the date difference is negative, 1 otherwise, -1 if parsing fails.
    """
    date = emailUtils.parsedate_tz(email_info["date"])
    last_received = email_info["received_hop_" + str(email_info["hop_count"])]
    last_received_list = re.split(r";", last_received)
    last_received_date = emailUtils.parsedate_tz(last_received_list[-1])
    if date is None or last_received_date is None:
        return -1
    try:
        emailUtils.mktime_tz(date)
        emailUtils.mktime_tz(last_received_date)
    except:
        return -1
    date_delta = int(
        (emailUtils.mktime_tz(last_received_date) - emailUtils.mktime_tz(date))
    )
    return 0 if date_delta < 0 else 1


def __get_missing_features(
    email_info: Dict[str, str], feature_list: Dict[str, int]
) -> None:
    """Checks for missing features in the email information.

    Args:
        email_info (Dict[str, str]): The email information dictionary.
        feature_list (Dict[str, int]): The dictionary to store missing features.
    """
    for name in email_info.keys():
        if "missing_" + name in FEATURES:
            feature_list["missing_" + name] = 1 if email_info[name] == "" else 0
    feature_list["missing_dmarc"] = 0


def __get_received_str_forged(email_info: Dict[str, str]) -> int:
    """Checks if any received headers contain the word 'forged'.

    Args:
        email_info (Dict[str, str]): The email information dictionary.

    Returns:
        int: 1 if 'forged' is found, 0 otherwise.
    """
    n = email_info["hop_count"]
    for i in range(1, n + 1):
        received = email_info["received_hop_" + str(i)]
        if "forged" in received:
            return 1
    return 0


def __check_if_valid(dict_to_check: Dict[str, str], str_val: str) -> bool:
    """Checks if a value is valid in a dictionary.

    Args:
        dict_to_check (Dict[str, str]): The dictionary to check.
        str_val (str): The value to check for.

    Returns:
        bool: True if the value is valid, False otherwise.
    """
    if dict_to_check is None:
        return False
    elif str_val not in dict_to_check:
        return False
    elif dict_to_check[str_val] is None:
        return False
    else:
        return True


def __get_for_domain_last_received(email_info: Dict[str, str]) -> str:
    """Extracts the domain from the last received header.

    Args:
        email_info (Dict[str, str]): The email information dictionary.

    Returns:
        str: The extracted domain or 'NA' if not found.
    """
    last_received_val = email_info.get("last_received", "")
    parsed_val = receivedParser.parse(last_received_val)
    if __check_if_valid(parsed_val, "envelope_for"):
        main_domain_parts = parsed_val["envelope_for"].split("@")[-1].split(".")
        if len(main_domain_parts) >= 2:
            main_domain = (
                main_domain_parts[-2] + "." + re.sub(r"\W+", "", main_domain_parts[-1])
            )
            return main_domain.lower()
    return "NA"


def __check_for_received_domain_equal(
    email_info: Dict[str, str], field_vals: List[str]
) -> int:
    """Checks if the domain from the last received header matches any given domains.

    Args:
        email_info (Dict[str, str]): The email information dictionary.
        field_vals (List[str]): The list of domains to check against.

    Returns:
        int: 1 if a match is found, 0 otherwise, -1 if no valid domain is found.
    """
    get_for_domain = __get_for_domain_last_received(email_info)
    if get_for_domain == "NA" or not field_vals:
        return -1
    return 1 if get_for_domain in field_vals else 0


def __get_from_domain_first_received(email_info: Dict[str, str]) -> List[str]:
    """Extracts domains from the first received header.

    Args:
        email_info (Dict[str, str]): The email information dictionary.

    Returns:
        List[str]: A list of extracted domains.
    """
    first_received_val = email_info["first_received"]
    parsed_val = receivedParser.parse(first_received_val)
    domains_list = []
    if __check_if_valid(parsed_val, "from_hostname"):
        if len(parsed_val["from_hostname"].split("@")) == 2:
            domains_list.append(parsed_val["from_hostname"].split("@")[-1])
    if __check_if_valid(parsed_val, "from_name"):
        if len(parsed_val["from_name"].split("@")) == 2:
            domains_list.append(parsed_val["from_name"].split("@")[-1])
    return domains_list


def __check_received_from_domain_equal(
    email_info: Dict[str, str], field_vals: List[str]
) -> int:
    """Checks if the domains from the first received header match any given domains.

    Args:
        email_info (Dict[str, str]): The email information dictionary.
        field_vals (List[str]): The list of domains to check against.

    Returns:
        int: 1 if a match is found, 0 otherwise, -1 if no valid domain is found.
    """
    domains_list_check = __get_from_domain_first_received(email_info)
    if not domains_list_check or not field_vals:
        return -1
    for item in field_vals:
        for item2 in domains_list_check:
            if item == item2:
                return 1
    return 0


def __get_features_array(email_info: Dict[str, str]) -> List[int]:
    """Extracts features from email information and returns them as an array.

    Args:
        email_info (Dict[str, str]): The email information dictionary.

    Returns:
        List[int]: An array of extracted features.
    """
    features_dict = {}
    features_dict["time_zone"] = __get_time_zone(email_info)
    features_dict["date_comp_date_received"] = __get_date_date_received_diff(email_info)
    __get_missing_features(email_info, features_dict)
    if (email_info["arc-authentication-results"] != "") and (
        "spf=none" not in email_info["arc-authentication-results"]
    ):
        features_dict["missing_received-spf"] = 0
    if (email_info["arc-authentication-results"] != "") and (
        "dmarc=none" not in email_info["arc-authentication-results"]
    ):
        features_dict["missing_dmarc"] = 0
    if (email_info["arc-authentication-results"] != "") and (
        "dkim=none" not in email_info["arc-authentication-results"]
    ):
        features_dict["missing_domainkey-signature"] = 0
    if email_info["dkim-signature"] != "":
        features_dict["missing_domainkey-signature"] = 0
    info_name = "content-transfer-encoding"
    features_names = ["str_content-encoding_empty"]
    conditions_check = [""]
    __get_str_features(
        email_info, info_name, features_dict, features_names, conditions_check
    )
    info_name = "from"
    features_names = ["str_from_question", "str_from_exclam", "str_from_chevron"]
    conditions_check = ["\\?", "!", "<.+>"]
    __get_str_features(
        email_info, info_name, features_dict, features_names, conditions_check
    )
    info_name = "to"
    features_names = ["str_to_chevron", "str_to_undisclosed", "str_to_empty"]
    conditions_check = ["<.+>", "Undisclosed Recipients", ""]
    __get_str_features(
        email_info, info_name, features_dict, features_names, conditions_check
    )
    info_name = "message-id"
    features_names = ["str_message-ID_dollar"]
    conditions_check = ["\\$"]
    __get_str_features(
        email_info, info_name, features_dict, features_names, conditions_check
    )
    info_name = "return-path"
    features_names = ["str_return-path_bounce", "str_return-path_empty"]
    conditions_check = ["bounce", ""]
    __get_str_features(
        email_info, info_name, features_dict, features_names, conditions_check
    )
    info_name = "reply-to"
    features_names = ["str_reply-to_question"]
    conditions_check = ["\\?"]
    __get_str_features(
        email_info, info_name, features_dict, features_names, conditions_check
    )
    info_name = "received-spf"
    features_names = [
        "str_received-SPF_bad",
        "str_received-SPF_softfail",
        "str_received-SPF_fail",
    ]
    conditions_check = ["bad", "softfail", "fail"]
    __get_str_features(
        email_info, info_name, features_dict, features_names, conditions_check
    )
    info_name = "content-type"
    features_names = ["str_content-type_texthtml"]
    conditions_check = ["text/html"]
    __get_str_features(
        email_info, info_name, features_dict, features_names, conditions_check
    )
    info_name = "precedence"
    features_names = ["str_precedence_list"]
    conditions_check = ["list"]
    __get_str_features(
        email_info, info_name, features_dict, features_names, conditions_check
    )
    info_name = "arc-authentication-results"
    features_names = [
        "str_received-SPF_bad",
        "str_received-SPF_softfail",
        "str_received-SPF_fail",
    ]
    conditions_check = ["spf=bad", "spf=softfail", "spf=fail"]
    __get_str_features(
        email_info, info_name, features_dict, features_names, conditions_check
    )
    info_name = "arc-authentication-results"
    features_names = ["str_dmarc_bad", "str_dmarc_softfail", "str_dmarc_fail"]
    conditions_check = ["dmarc=bad", "dmarc=softfail", "dmarc=fail"]
    __get_str_features(
        email_info, info_name, features_dict, features_names, conditions_check
    )
    info_name = "arc-authentication-results"
    features_names = ["str_dkim_bad", "str_dkim_softfail", "str_dkim_fail"]
    conditions_check = ["dkim=bad", "dkim=softfail", "dkim=fail"]
    __get_str_features(
        email_info, info_name, features_dict, features_names, conditions_check
    )
    features_dict["received_str_forged"] = __get_received_str_forged(email_info)
    from_emails = __extract_emails(email_info["from"])
    reply_to_emails = __extract_emails(email_info["reply-to"])
    features_dict["email_match_from_reply-to"] = __email_same_check(
        from_emails, reply_to_emails
    )
    message_id_domains = __extract_domains(email_info["message-id"])
    from_domains = __extract_domains(email_info["from"])
    return_path_domains = __extract_domains(email_info["return-path"])
    sender_domains = __extract_domains(email_info["sender"])
    reply_to_domains = __extract_domains(email_info["reply-to"])
    to_domains = __extract_domains(email_info["to"])
    in_reply_to_domains = __extract_domains(email_info["in-reply-to"])
    errors_to_domains = __extract_domains(email_info["errors-to"])
    references_domains = __extract_domains(email_info["references"])
    features_dict["domain_val_message-id"] = __check_shady_domain(message_id_domains)
    features_dict["domain_match_message-id_from"] = __domain_match_check(
        message_id_domains, from_domains
    )
    features_dict["domain_match_from_return-path"] = __domain_match_check(
        from_domains, return_path_domains
    )
    features_dict["domain_match_message-id_return-path"] = __domain_match_check(
        message_id_domains, return_path_domains
    )
    features_dict["domain_match_message-id_sender"] = __domain_match_check(
        message_id_domains, sender_domains
    )
    features_dict["domain_match_message-id_reply-to"] = __domain_match_check(
        message_id_domains, reply_to_domains
    )
    features_dict["domain_match_return-path_reply-to"] = __domain_match_check(
        return_path_domains, reply_to_domains
    )
    features_dict["domain_match_reply-to_to"] = __domain_match_check(
        reply_to_domains, to_domains
    )
    features_dict["domain_match_to_in-reply-to"] = __domain_match_check(
        to_domains, in_reply_to_domains
    )
    features_dict["domain_match_errors-to_message-id"] = __domain_match_check(
        errors_to_domains, message_id_domains
    )
    features_dict["domain_match_errors-to_from"] = __domain_match_check(
        errors_to_domains, from_domains
    )
    features_dict["domain_match_errors-to_sender"] = __domain_match_check(
        errors_to_domains, sender_domains
    )
    features_dict["domain_match_errors-to_reply-to"] = __domain_match_check(
        errors_to_domains, reply_to_domains
    )
    features_dict["domain_match_sender_from"] = __domain_match_check(
        sender_domains, from_domains
    )
    features_dict["domain_match_references_reply-to"] = __domain_match_check(
        references_domains, reply_to_domains
    )
    features_dict["domain_match_references_in-reply-to"] = __domain_match_check(
        references_domains, in_reply_to_domains
    )
    features_dict["domain_match_references_to"] = __domain_match_check(
        references_domains, to_domains
    )
    features_dict["domain_match_from_reply-to"] = __domain_match_check(
        from_domains, reply_to_domains
    )
    features_dict["domain_match_to_from"] = __domain_match_check(
        to_domains, from_domains
    )
    features_dict["domain_match_to_message-id"] = __domain_match_check(
        to_domains, message_id_domains
    )
    features_dict["domain_match_to_received"] = __check_for_received_domain_equal(
        email_info, to_domains
    )
    features_dict["domain_match_reply-to_received"] = __check_for_received_domain_equal(
        email_info, reply_to_emails
    )
    features_dict["domain_match_from_received"] = __check_received_from_domain_equal(
        email_info, from_domains
    )
    features_dict["domain_match_return-path_received"] = (
        __check_received_from_domain_equal(email_info, return_path_domains)
    )
    features_array = [features_dict[feature] for feature in features_dict.keys()]
    return features_array


def __get_email_info(email_path: str) -> Union[Dict[str, str], int]:
    """Reads and parses email information from a file.

    Args:
        email_path (str): The path to the email file.

    Returns:
        Union[Dict[str, str], int]: The parsed email information dictionary or -1 if an error occurs.
    """
    email_dict = {column: "" for column in HEADER_INFORMATION}
    try:
        with open(email_path, "r", encoding="latin_1") as email_file:
            email = email_file.read()
    except Exception as e:
        logging.error(f"Error reading email file {email_path}: {e}")
        return -1
    header = parser.parsestr(email)
    features_lower_case = [x.lower() for x in header.keys()]
    received_list = header.get_all("received")
    hops = len(received_list) if received_list else 0
    for i, received_field in enumerate(received_list or []):
        email_dict[f"received_hop_{i + 1}"] = received_field
    if received_list:
        email_dict["first_received"] = received_list[0]
        email_dict["last_received"] = received_list[-1]
    temp_dict = dict(zip(features_lower_case, header.values()))
    for key in temp_dict.keys():
        if key in HEADER_INFORMATION:
            email_dict[key] = temp_dict[key]
    email_dict["hop_count"] = hops
    return email_dict


def __process_email(
    line: str, email_cache: Dict[str, List[int]], label_dict: Dict[str, int]
) -> Tuple[Union[List[int], None], Union[int, None], Union[str, None]]:
    """Processes a single email line to extract features and label.

    Args:
        line (str): The email line to process.
        email_cache (Dict[str, List[int]]): A cache to store processed email features.
        label_dict (Dict[str, int]): A dictionary mapping labels to integers.

    Returns:
        Tuple[Union[List[int], None], Union[int, None], Union[str, None]]: The extracted features, label, and email path.
    """
    label, email_path = line.split(" ")
    if email_path not in email_cache:
        email_info = __get_email_info(email_path)
        if email_info == -1:
            return None, None, None
        email_cache[email_path] = __get_features_array(email_info)
    return email_cache[email_path], label_dict[label], email_path


def get_training_test_set(
    index_path: str, values: List[str], percent: float
) -> Tuple[List[List[int]], List[int]]:
    """Generates training and test sets from the given index file.

    Args:
        index_path (str): The path to the index file.
        values (List[str]): The list of values to include.
        percent (float): The percentage of data to use.

    Returns:
        Tuple[List[List[int]], List[int]]: The training set and labels.
    """
    if not os.path.exists(index_path):
        logging.info("Index file not found. Creating index using dataOrganizer...")
        organize_data()
    with open(index_path, "r", encoding="latin_1") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if line.split(" ")[0] in values]
    random.shuffle(lines)
    num_samples = int(len(lines) * percent)
    selected_lines = lines[:num_samples]
    train_set, labels = [], []
    label_dict = {"ham": 0, "spam": 1, "phishing": 1}
    email_cache = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(__process_email, line, email_cache, label_dict)
            for line in selected_lines
        ]
        for future in concurrent.futures.as_completed(futures):
            features, label, _ = future.result()
            if features is not None and label is not None:
                train_set.append(features)
                labels.append(label)
    return train_set, labels


def get_example_test_set(
    index_path: str,
) -> Tuple[List[List[int]], List[int], List[str]]:
    """Generates an example test set from the given index file.

    Args:
        index_path (str): The path to the index file.

    Returns:
        Tuple[List[List[int]], List[int], List[str]]: The test set, labels, and email paths.
    """
    with open(index_path, "r", encoding="latin_1") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    test_set, labels, email_paths = [], [], []
    label_dict = {"ham": 0, "spam": 1, "phishing": 1}
    email_cache = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(__process_email, line, email_cache, label_dict)
            for line in lines
        ]
        for future in concurrent.futures.as_completed(futures):
            features, label, email_path = future.result()
            if features is not None and label is not None:
                test_set.append(features)
                labels.append(label)
                email_paths.append(email_path)
    return test_set, labels, email_paths


def main() -> None:
    """Main function to start the data extraction process."""
    logging.info("Starting data extraction process...")
    index_path = "Dataset/index"
    values = ["ham", "phishing"]
    percent = 1.0
    train_set, labels = get_training_test_set(index_path, values, percent)
    logging.info(f"Training set size: {len(train_set)}, Labels size: {len(labels)}")


if __name__ == "__main__":
    main()
