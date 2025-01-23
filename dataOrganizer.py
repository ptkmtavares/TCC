import os
import shutil
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
from config import (
    LOG_FORMAT,
    DATA_DIR,
    INDEX_PATH,
    SPAM_HAM_DATA_DIR,
    SPAM_HAM_INDEX_PATH,
    PHISHING_DIR,
)

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def __setup_directories(data_path: str, index_path: str) -> None:
    """Sets up the necessary directories for data and index files.

    Args:
        data_path (str): The path to the data directory.
        index_path (str): The path to the index directory.
    """
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)


def __process_email_file(file_path: str, encoding: str = "latin_1") -> str:
    """Reads the content of an email file.

    Args:
        file_path (str): The path to the email file.
        encoding (str, optional): The encoding of the email file. Defaults to 'latin_1'.

    Returns:
        str: The content of the email file.
    """
    try:
        with open(file_path, "r", encoding=encoding) as file:
            return file.read()
    except UnicodeDecodeError:
        logging.error(f"Unicode error reading file: {file_path}")
        return ""


def __get_phishing_email_list(folder_path: str) -> List[str]:
    """Gets a list of phishing emails from a folder.

    Args:
        folder_path (str): The path to the folder containing phishing emails.

    Returns:
        List[str]: A list of phishing email contents.
    """
    email_list = []

    def __process_file(filename: str) -> List[str]:
        """Processes a single file and returns a list of email contents.

        Args:
            filename (str): The name of the file to process.

        Returns:
            List[str]: A list of email contents.
        """
        if not filename.endswith(".txt"):
            return []

        file_path = os.path.join(folder_path, filename)
        content = __process_email_file(file_path)
        if not content:
            return []

        splits = content.split("\nFrom jose@monkey.org")
        return [
            f"From jose@monkey.org{split}" if i > 0 else split
            for i, split in enumerate(splits)
        ]

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(__process_file, filename)
            for filename in os.listdir(folder_path)
        ]
        for future in futures:
            email_list.extend(future.result())

    return email_list


def __copy_spam_ham_files(source_path: str, dest_path: str) -> None:
    """Copies spam and ham email files from the source to the destination.

    Args:
        source_path (str): The source directory path.
        dest_path (str): The destination directory path.
    """
    logging.info(f"Copying files from {source_path} to {dest_path}")

    def copy_file(filename: str) -> None:
        src = os.path.join(source_path, filename)
        dst = os.path.join(dest_path, filename)
        shutil.copy(src, dst)

    with ThreadPoolExecutor() as executor:
        list(executor.map(copy_file, os.listdir(source_path)))


def __update_index_file(
    index_path: str, data_folder_path: str, spam_ham_index_path: str
) -> None:
    """Updates the index file with the new data folder path.

    Args:
        index_path (str): The path to the index file.
        data_folder_path (str): The new data folder path.
        spam_ham_index_path (str): The path to the spam and ham index file.
    """
    logging.info("Updating index file")

    with open(spam_ham_index_path, "r", encoding="latin_1") as file:
        index_text = file.read()

    index_text = index_text.replace("../data", data_folder_path)

    with open(index_path, "w", encoding="latin_1") as index:
        index.write(index_text)


def __write_phishing_emails(
    emails: List[str], data_path: str, index_path: str, start_index: int
) -> None:
    """Writes phishing emails to the data directory and updates the index file.

    Args:
        emails (List[str]): The list of phishing email contents.
        data_path (str): The path to the data directory.
        index_path (str): The path to the index file.
        start_index (int): The starting index for the email filenames.
    """
    logging.info(f"Processing {len(emails)} phishing emails")

    def __write_email(args: Tuple[int, str]) -> str:
        """Writes a single phishing email to a file.

        Args:
            args (Tuple[int, str]): A tuple containing the index and email content.

        Returns:
            str: The index entry for the written email.
        """
        i, email = args
        filename = f"inmail.{i + start_index}"
        filepath = os.path.join(data_path, filename)

        with open(filepath, "w", encoding="latin_1") as f:
            f.write(email)

        return f"phishing {filepath}\n"

    with ThreadPoolExecutor() as executor:
        index_entries = list(executor.map(__write_email, enumerate(emails)))

    with open(index_path, "a", encoding="latin_1") as index:
        index.writelines(index_entries)


def main():
    """Main function to organize the dataset."""
    data_folder_path = DATA_DIR
    index_path = INDEX_PATH
    spam_ham_data_path = SPAM_HAM_DATA_DIR
    spam_ham_index_path = SPAM_HAM_INDEX_PATH
    phishing_folder_path = PHISHING_DIR

    __setup_directories(data_folder_path, index_path)

    __copy_spam_ham_files(spam_ham_data_path, data_folder_path)
    __update_index_file(index_path, data_folder_path, spam_ham_index_path)

    phishing_emails = __get_phishing_email_list(phishing_folder_path)
    __write_phishing_emails(
        phishing_emails, data_folder_path, index_path, start_index=75420
    )

    logging.info("Processamento concluído com sucesso!")


if __name__ == "__main__":
    main()
