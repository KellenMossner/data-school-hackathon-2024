import json
import sys
from shapely.geometry import Polygon

def calculate_L1_size(json_data):
    data = json.loads(json_data)
    L1 = next(box for box in data['boxes'] if box['label'] == 'L1')
    # Extract the points of the pothole polygon
    points = L1['points']
    
    # Create a Shapely polygon
    poly = Polygon(points)
    
    # Calculate the area of the polygon
    area = poly.area
    
    # Calculate the perimeter of the polygon
    perimeter = poly.length
    
    # Get the bounding box dimensions
    width = float(L1['width'])
    height = float(L1['height'])
    
    # Calculate the aspect ratio
    aspect_ratio = width / height
    
    # Calculate the percentage of the image occupied by the pothole
    image_width = data['width']
    image_height = data['height']
    image_area = image_width * image_height
    pothole_percentage = (area / image_area) * 100
    
    return {
        'area': area,
        'perimeter': perimeter,
        'width': width,
        'height': height,
        'aspect_ratio': aspect_ratio,
        'percentage_of_image': pothole_percentage
    }


def calculate_pothole_size(json_data):
    # Parse the JSON data
    data = json.loads(json_data)
    
    # Find the pothole polygon
    pothole = next(box for box in data['boxes'] if box['label'] == 'pothole')
    
    # Extract the points of the pothole polygon
    points = pothole['points']
    
    # Create a Shapely polygon
    poly = Polygon(points)
    
    # Calculate the area of the polygon
    area = poly.area
    
    # Calculate the perimeter of the polygon
    perimeter = poly.length
    
    # Get the bounding box dimensions
    width = float(pothole['width'])
    height = float(pothole['height'])
    
    # Calculate the aspect ratio
    aspect_ratio = width / height
    
    # Calculate the percentage of the image occupied by the pothole
    image_width = data['width']
    image_height = data['height']
    image_area = image_width * image_height
    pothole_percentage = (area / image_area) * 100
    
    return {
        'area': area,
        'perimeter': perimeter,
        'width': width,
        'height': height,
        'aspect_ratio': aspect_ratio,
        'percentage_of_image': pothole_percentage
    }

def main():
    # Check if a file path is provided as a command-line argument
    if len(sys.argv) < 2:
        print("Please provide the path to the JSON file as a command-line argument.")
        sys.exit(1)
    
    # Get the file path from the command-line argument
    file_path = sys.argv[1]
    
    try:
        # Read the JSON data from the file
        with open(file_path, 'r') as file:
            json_data = file.read()
        
        # Calculate the pothole size
        result = calculate_pothole_size(json_data)
        
        # Print the results
        print("Pothole Size Calculations:")
        for key, value in result.items():
            print(f"{key.replace('_', ' ').title()}: {value:.2f}")

        print('\n')
        # Calculate the pothole size
        result = calculate_L1_size(json_data)
        
        # Print the results
        print("L1 Size Calculations:")
        for key, value in result.items():
            print(f"{key.replace('_', ' ').title()}: {value:.2f}")
    
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' does not contain valid JSON data.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()