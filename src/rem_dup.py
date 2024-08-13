import json
import os

def process_json_file(file_path):
    # Read the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Find all L1 items
    l1_items = [item for item in data if item['name'] == 'L1']
    
    # If there's more than one L1 item, remove the larger one
    if len(l1_items) > 1:
        # Calculate the area of each L1 item
        for item in l1_items:
            box = item['box']
            item['area'] = (box['x2'] - box['x1']) * (box['y2'] - box['y1'])
        
        # Sort L1 items by area in descending order
        l1_items.sort(key=lambda x: x['area'], reverse=True)
        
        # Remove the largest L1 item
        data.remove(l1_items[0])
    
    # Write the updated data back to the file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def process_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            process_json_file(file_path)
            print(f"Processed: {filename}")

def main():
    print("Removing duplicate: \n[L1]\nTRAIN json files\nkeeping smallest one")
    directory_path = 'data/cv_train_out'
    process_directory(directory_path)
    print("Done!")


if __name__ == "__main__":
    main()