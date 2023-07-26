import pandas as pd
import pickle


def load_datasets():
    APKs_Combined_Processed_csv = "I:/azoo-datasets/LVDAndro_APKs_Combined_Processed.csv"
    vulnerability_df = pd.read_csv(APKs_Combined_Processed_csv,low_memory=False).fillna("")
    print(vulnerability_df.columns)
    print(vulnerability_df.CWE_ID.value_counts())

    return vulnerability_df

def process_df(vulnerability_df):
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace(
        "CWE-532 Insertion of Sensitive Information into Log File", "CWE-532")
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-532:", "CWE-532")

    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-276 Incorrect Default Permissions", "CWE-276")
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-276:", "CWE-276")

    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace(
        "CWE-312 Cleartext Storage of Sensitive Information", "CWE-312")
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-312:", "CWE-312")

    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-200:", "CWE-200")
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-200 Information Exposure", "CWE-200")

    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace(
        "CWE-89 Improper Neutralization of Special Elements used in an SQL Command ('SQL Injection')", "CWE-89")
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-89:", "CWE-89")

    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-749 Exposed Dangerous Method or Function",
                                                                    "CWE-749")
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-749:", "CWE-749")

    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace(
        "CWE-327 Use of a Broken or Risky Cryptographic Algorithm", "CWE-327")
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-327:", "CWE-327")

    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-919 - Weaknesses in Mobile Applications",
                                                                    "CWE-919")
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-919:", "CWE-919")

    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-295 Improper Certificate Validation",
                                                                    "CWE-295")
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-295:", "CWE-295")

    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace(
        "CWE-649 Reliance on Obfuscation or Encryption of Security-Relevant Inputs without Integrity Checking",
        "CWE-649")
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-649:", "CWE-649")

    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-250 Execution with Unnecessary Privileges",
                                                                    "CWE-250")
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-250:", "CWE-250")

    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-330:", "CWE-330")
    vulnerability_df['CWE_ID'] = vulnerability_df['CWE_ID'].replace("CWE-330 Use of Insufficiently Random Values",
                                                                    "CWE-330")


