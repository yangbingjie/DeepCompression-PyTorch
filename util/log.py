import os

unit_switcher = {
    'B': 1,  # byte
    'K': 1024,  # kilo byte
    'M': 1048576,  # million byte
}


# log the file size
# unit: "B" or "KB" or "M"
def log_file_size(path, unit='B'):
    if unit in unit_switcher:
        print('The file size is', str(get_file_size(path, unit)) + ' ' + unit)
    else:
        print('Error input for unit:', unit)


# get the file size
def get_file_size(path, unit='B'):
    size = os.path.getsize(path) / unit_switcher[unit]
    return round(size, 2)
