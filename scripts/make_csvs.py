# scripts/extract_pdfs.py
import pdfplumber, re, os, csv
from glob import glob
from tqdm import tqdm

PDF_DIR = "ATTESTATIONS SALARIES DECLARES"
OUT_DIR = "extracted_csvs"
os.makedirs(OUT_DIR, exist_ok=True)

def extract_workers_from_pdf(pdf_path):
    workers = []
    # try to infer month-year from header lines if present
    # we will fallback to None if not found
    month = None
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                full_text += txt + "\n"

    # Try to capture "mois année" lines like "décembre 2023" and take first occurrence
    m = re.search(r"(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+(\d{4})", full_text, re.IGNORECASE)
    if m:
        month = f"{m.group(1).lower()} {m.group(2)}"

    # Parse worker lines
    lines = full_text.splitlines()
    for line in lines:
        line = line.strip()
        if not re.match(r"^\d{9}\b", line):
            continue
        # attempt to match "imm name days salary"
        match = re.match(r"^(\d{9})\s+([A-ZÉÈÊÂÎÔÛÀÙÏÜÇ\-\s]+?)\s+(\d{1,2})\s+([\d\s,.]+)$", line)
        if not match:
            continue
        imm, name, days, salary_raw = match.groups()
        name = " ".join(name.split()).title()
        salary_clean = salary_raw.replace(" ", "").replace(",", ".")
        num = re.findall(r"\d+\.?\d*", salary_clean)
        if not num:
            continue
        salary_value = float(num[0])
        workers.append({
            "immatriculationNumber": imm,
            "full_name": name,
            "nombre_jours": int(days),
            "salaire": salary_value,
            "month": month
        })
    return workers

def main():
    pdfs = glob(os.path.join(PDF_DIR, "*.pdf"))
    for p in tqdm(pdfs):
        idname = os.path.splitext(os.path.basename(p))[0]
        workers = extract_workers_from_pdf(p)
        out_csv = os.path.join(OUT_DIR, f"{idname}_workers.csv")
        if workers:
            # add ID_adherent field
            for w in workers:
                w["ID_adherent"] = idname
            keys = ["ID_adherent","month","immatriculationNumber","full_name","nombre_jours","salaire"]
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for w in workers:
                    writer.writerow({k: w.get(k) for k in keys})

if __name__ == "__main__":
    main()
