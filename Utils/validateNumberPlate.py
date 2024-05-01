import re

def validate(plate_number):
    pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{4}$'

    plate_number = "JH09AE6854"

    match = re.match(pattern, plate_number)

    if match:
        print("Valid Indian vehicle registration plate number:", plate_number)
    else:
        print("Invalid Indian vehicle registration plate number:", plate_number)
