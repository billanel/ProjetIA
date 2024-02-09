
import json
import csv
import os

def json_to_csv(input_file):
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
           
        if isinstance(data, list):
            # Cas où le fichier JSON contient une liste d'objets
            fieldnames = set()
            for row in data:
                fieldnames.update(row.keys())
        elif isinstance(data, dict):
            # Cas où le fichier JSON contient un objet simple
            fieldnames = data.keys()
        else:
            print("Format de fichier JSON non pris en charge.")
            return None
           
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print("Erreur lors de la lecture du fichier JSON :", e)
        return None
   
    output_folder = os.path.join(os.path.expanduser("~"), "Downloads", "out")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    csv_file = os.path.join(output_folder, "output.csv")
   
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
       
        if isinstance(data, list):
            # Cas où le fichier JSON contient une liste d'objets
            for row in data:
                # Assurez-vous que chaque ligne a toutes les colonnes nécessaires
                for fieldname in fieldnames:
                    if fieldname not in row:
                        row[fieldname] = ""
                writer.writerow(row)
        elif isinstance(data, dict):
            # Cas où le fichier JSON contient un objet simple
            writer.writerow(data)
   
    return csv_file

# Chemin vers le fichier JSON dans le dossier d'entrée
input_file = "C:\\Users\\HP\\Downloads\\sample2.json"

# Convertir le fichier JSON en fichier CSV
csv_file = json_to_csv(input_file)
if csv_file:
    print(f"Le fichier CSV a été généré avec succès dans : {csv_file}")
else:
    print("Une erreur s'est produite lors de la génération du fichier CSV.")
