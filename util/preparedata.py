import pandas as pd

# CSV-Datei laden
input_file = 'input.csv'  # Name der Eingabedatei
output_file = 'output.csv'  # Name der Ausgabedatei

# Daten einlesen
data = pd.read_csv(input_file, header=None)

# Überprüfen, ob die neuesten Daten zuerst sind und die Reihenfolge umkehren
# Annahme: Spalte 0 enthält ein Datum oder etwas Ähnliches, das für die Reihenfolge spricht
# Wenn dies nicht der Fall ist, kannst du diese Zeile ignorieren.
# if pd.to_datetime(data[0], errors='coerce').is_monotonic_decreasing:
data = data.iloc[::-1]  # Reihenfolge umkehren, falls neue Daten zuerst kommen

# Umwandlung der Werte
def convert_value(value):
    if isinstance(value, str):
        # Entferne unerwünschte Zeichen und konvertiere in float
        value = value.replace('.','').replace('"', '').replace(',', '.').replace('%', '').strip()
        try:
            return float(value)
        except ValueError:
            return None  # Falls die Umwandlung fehlschlägt, None zurückgeben
    return value

# Wende die Umwandlung auf die relevanten Spalten an (1: Open, 2: Close, 3: High, 4: Low)
for col in [1, 2, 3, 4]:
    data[col] = data[col].apply(convert_value)

# Speichern der bereinigten Daten in eine neue CSV-Datei
data[[1, 2, 3, 4]].to_csv(output_file, header=False, index=False)

print(f'Die umgewandelte Datei wurde als "{output_file}" gespeichert.')
