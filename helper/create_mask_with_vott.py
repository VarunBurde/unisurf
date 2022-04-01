import os
import argparse
import json
import numpy as np
import cv2

base_path = os.getcwd()

def parse_args():
    parser = argparse.ArgumentParser(description=" get the mask from vott style json ")

    parser.add_argument("--images", default="images", help="input path to the images")
    parser.add_argument("--json_file", default="somthing", help="input to json file")
    parser.add_argument("--out", default="mask_folder", help="mask_folder")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    json_file_path = args.json_file
    output_path = args.out
    f = open(json_file_path)
    data = json.load(f)

    for asset in data["assets"]:
        contour = []
        name = data["assets"][asset]['asset']['name']
        path = data["assets"][asset]['asset']['path']
        height = data["assets"][asset]['asset']['size']['height']
        width = data["assets"][asset]['asset']['size']['width']
        points = data["assets"][asset]['regions'][0]['points']
        mask_name = name[-7:]
        mask_path = os.path.join(output_path,mask_name)
        for i in range(len(points)):
            p = [ int(points[i]['x']), int(points[i]['y'])]
            contour.append(p)
        contour = np.array(contour)
        contour = np.expand_dims(contour,axis=1)
        img = np.zeros([height,width,3],dtype=np.uint8)
        img = cv2.drawContours(img, [contour], 0, (255, 255, 255), -1)
        cv2.imwrite(mask_path, img)


