import pdfplumber
import re

def extract_all_workers(pdf_path):
    workers = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:

            # Try to extract the text normally
            text = page.extract_text()
            if not text:
                continue

            # Normalize spaces
            text = re.sub(r"[ ]{2,}", " ", text)

            # Extract only lines that look like "ID Name Days Salary"
            # Example line:
            # 102392190 BAHAZZAZ FATIMA ZAHRA 26 4189,94
            for line in text.split("\n"):
                line = line.strip()

                # CNSS lines always start with a 9-digit immatriculation number
                if not re.match(r"^\d{9}\b", line):
                    continue

                # Regex for one complete line
                match = re.match(
                    r"^(\d{9})\s+([A-ZÉÈÊÂÎÔÛÀÙÏÜÇ ]+?)\s+(\d{1,2})\s+([\d\s,.]+)$",
                    line
                )

                if not match:
                    continue

                imm, name, days, salary_raw = match.groups()

                # Clean name
                name = " ".join(name.split()).title()

                # Clean salary
                salary_clean = salary_raw.replace(" ", "").replace(",", ".")
                salary_value = float(re.findall(r"\d+\.?\d*", salary_clean)[0])

                workers.append({
                    "immatriculationNumber": imm,
                    "full_name": name,
                    "nombre_jours": int(days),
                    "salaire": salary_value
                })

    return workers


# TEST
pdf_path = "ATTESTATIONS SALARIES DECLARES/67.pdf"
workers = extract_all_workers(pdf_path)

print("COUNT =", len(workers))
print(workers[:5])
