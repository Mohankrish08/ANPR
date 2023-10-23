# installing libraries
import easyocr
import csv
import datetime



# Initialize the OCR reader
reader = easyocr.Reader(['en'], gpu=False)


# reading the license palte
def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)
    extracted_text = ""

    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')
        extracted_text += text

    return extracted_text

# Function to check if a license plate complies with the expected format
def license_complies_format(text):
    if len(text) == 10:
        first_two_letters = text[:2]
        next_two_digits = text[2:4]
        next_two_letters = text[4:6]
        last_four_digits = text[6:]

        if (first_two_letters.isalpha() and next_two_digits.isdigit()) or (next_two_letters.isalpha() and last_four_digits.isdigit()):
            return True
    return False

# Function to write results to a CSV file
# ...

def write_csv(results, output_path):
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['license_plate', 'timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        unique_plates_set = set()

        for result in results:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for plate in result['plate_unique']:
                if plate not in unique_plates_set:
                    unique_plates_set.add(plate)  # Add the license plate to the set
                    writer.writerow({'license_plate': plate, 'timestamp': timestamp})
# ...
