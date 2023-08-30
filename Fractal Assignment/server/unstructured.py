import csv

csv_file_path = 'employee_data.csv'

def get_headers(file_path):
    with open(file_path, 'r') as file:
      csv_reader = csv.reader(file)
      header = next(csv_reader)
    return header

# print("Header information:")
# print(header)

# print("\nIndividual column names:")
# for column_name in header:
#     print(column_name)

if __name__ == '__main__':
  print(get_headers(csv_file_path))