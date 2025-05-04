import json
test_dict = {"operators": [
  {
    "color": 0,
    "id": 0,
    "type": 0,
    "init_hex": 0},
  {
    "color": 0,
    "id": 1,
    "type": 1,
    "init_hex": 1},
  {
    "color": 0,
    "id": 2,
    "type": 2,
    "init_hex": 2
  },
  {
    "color": 1,
    "id": 10,
    "type": 0,
    "init_hex": 6674},
  {
    "color": 1,
    "id": 11,
    "type": 1,
    "init_hex": 6675},
  {
    "color": 1,
    "id": 12,
    "type": 2,
    "init_hex": 6676
  }]}
json_str = json.dumps(test_dict)
def write_json():
  with open("5.json", "w") as f:
       json.dump(test_dict, f)
       print("done...")

def read_json():
  with open("2.json", 'r') as load_f:
       load_dict = json.load(load_f)
       print(load_dict)
       print(load_dict.type)

write_json()