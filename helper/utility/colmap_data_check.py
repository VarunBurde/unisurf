#!/usr/bin/env python3

import argparse
import numpy as np
import math
import cv2
import os

def parse_args():
	parser = argparse.ArgumentParser(description="convert a text colmap data to unisurf camera matrix")

	parser.add_argument("--images", default="images", help="input path to the images")
	parser.add_argument("--output", default="output", help='path to output')

	args = parser.parse_args()
	return args



def qvec2rotmat(qvec):
	return np.array([
		[
			1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
			2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
			2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
		], [
			2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
			1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
			2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
		], [
			2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
			2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
			1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
		]
	])

def rotmat(a, b):
	a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
	v = np.cross(a, b)
	c = np.dot(a, b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))


if __name__ == "__main__":
    args = parse_args()
    IMAGE_FOLDER=args.images
    TEXT_FOLDER= os.path.dirname(IMAGE_FOLDER)
    OUT_PATH=args.output
    print(f"outputting to {OUT_PATH}...")
    with open(os.path.join(TEXT_FOLDER,"cameras.txt"), "r") as f:
        angle_x=math.pi/2
        for line in f:
            # 1 SIMPLE_RADIAL 2048 1536 1580.46 1024 768 0.0045691
            # 1 OPENCV 3840 2160 3178.27 3182.09 1920 1080 0.159668 -0.231286 -0.00123982 0.00272224
            # 1 RADIAL 1920 1080 1665.1 960 540 0.0672856 -0.0761443
            if line[0]=="#":
                continue
            els=line.split(" ")
            w = float(els[2])
            h = float(els[3])
            fl_x = float(els[4])
            fl_y = float(els[4])
            k1 = 0
            k2 = 0
            p1 = 0
            p2 = 0
            cx = w / 2
            cy = h / 2
            if (els[1]=="SIMPLE_RADIAL"):
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
            elif (els[1]=="RADIAL"):
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
                k2 = float(els[8])
            elif (els[1]=="OPENCV"):
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
                k1 = float(els[8])
                k2 = float(els[9])
                p1 = float(els[10])
                p2 = float(els[11])
            else:
                print("unknown camera model ", els[1])
            # fl = 0.5 * w / tan(0.5 * angle_x);
            angle_x= math.atan(w/(fl_x*2))*2
            angle_y= math.atan(h/(fl_y*2))*2
            fovx=angle_x*180/math.pi
            fovy=angle_y*180/math.pi

    print(f"camera:\n\tres={w,h}\n\tcenter={cx,cy}\n\tfocal={fl_x,fl_y}\n\tfov={fovx,fovy}\n\tk={k1,k2} p={p1,p2} ")

    with open(os.path.join(TEXT_FOLDER,"images.txt"), "r") as f:
        i=0
        bottom = np.array([0,0,0,1.]).reshape([1,4])
        out={
            "camera_angle_x":angle_x,
            "camera_angle_y":angle_y,
            "fl_x":fl_x,
            "fl_y":fl_y,
            "k1":k1,
            "k2":k2,
            "p1":p1,
            "p2":p2,
            "cx":cx,
            "cy":cy,
            "w":w,
            "h":h,
            "frames":[]
        }

        up=np.zeros(3)
        for line in f:
            line=line.strip()
            if line[0]=="#":
                continue
            i=i+1
            if  i%2==1 :
                elems=line.split(" ") # 1-4 is quat, 5-7 is trans, 9 is filename
                #name = str(PurePosixPath(Path(IMAGE_FOLDER, elems[9])))
                # why is this requireing a relitive path while using ^
                image_rel = os.path.relpath(IMAGE_FOLDER)
                name = str(f"./{image_rel}/{elems[9]}")
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                R = qvec2rotmat(-qvec)
                t = tvec.reshape([3,1])
                c2w = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
                frame={"file_path":name,"transform_matrix": c2w, "image_id": image_id}
                out["frames"].append(frame)

    print(" making 3d point dict")
    ##### Normalization using 3d points
    three_d_points = {}
    ID = []

    with open(os.path.join(TEXT_FOLDER,"points3D.txt"), "r") as f:
        for line in f:
            if line[0]=="#":
                continue
            els = line.split(" ")
            X = float(els[1])
            Y = float(els[2])
            Z = float(els[3])
            point = [X, Y, Z]
            low_x = low_y = 0
            high_x = high_y = 0
            for i in range(8, len(els), 2):
                ID = int(els[i])
                if not ID in three_d_points.keys():
                    three_d_points[ID] = []
                three_d_points[ID].append(point)
                i += 2

    # for key in three_d_points:
    #     print(key, len(three_d_points[key]))

    intrinsic = np.eye(4, dtype=np.float64)
    intrinsic[0, 0] = 2 * (fl_x / w)
    intrinsic[1, 1] = 2 * (fl_y / h)
    intrinsic[0, 2] = 0
    intrinsic[1, 2] = 0
    print("intrinsics")
    print(intrinsic)

    cameras = {}
    i =0
    for f in out["frames"]:
        # f["transform_matrix"][0:3,3] *= 4.0 / avglen # scale to "nerf sized"
        extrinsic =  f["transform_matrix"]
        cameras["world_mat_%d" % i] = extrinsic
        cameras["camera_mat_%d" % i] = intrinsic
        low_x = low_y = 0
        high_x = high_y = 0
        for point in three_d_points[f['image_id']]:
            point = np.array(point)
            point = np.append(point,1)
            transformation_overall = np.matmul(intrinsic , extrinsic)
            image_cordinates = np.matmul(transformation_overall,point)
            image_cordinates = image_cordinates / image_cordinates[2]
            image_cordinates = image_cordinates[0:2]

            ### To check if point are in range -1 to 1
            if image_cordinates[0] < low_x:
                low_x = image_cordinates[0]
            if image_cordinates[1] < low_y:
                low_y = image_cordinates[1]
            if image_cordinates[0] > high_x:
                high_x = image_cordinates[0]
            if image_cordinates[1] > high_y:
                high_y = image_cordinates[1]
            # print(image_cordinates)

        print(low_x, low_y)
        print(high_x, high_y)
        print("next camera")

        #### saving to file
        mat_path = OUT_PATH
        img_dir = os.path.join(mat_path, "image")
        if not os.path.exists(img_dir):
            os.mkdir(os.path.join(mat_path, "image"))
        img = cv2.imread(f["file_path"])
        img_path = os.path.join(mat_path, 'image','{0:06}.png'.format(i))
        cv2.imwrite(img_path, img)
        np.savez(os.path.join(mat_path, 'cameras.npz'), **cameras)
        print("saving", '{0:06}.png'.format(i))
        i = i +1



